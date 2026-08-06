"""Training script for the Meta-Classifier Fusion Engine.

Collects predictions from spatial, frequency, and semantic streams
on the training set and trains the XGBoost + LightGBM ensemble
with Isotonic Regression calibration.

Usage:
    python -m training.train_fusion --data_dir data/processed
"""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image

from streams.spatial.model import SpatialForensicsStream
from streams.frequency.model import FrequencyForensicsStream
from streams.frequency.preprocess import prepare_freq_tensor
from streams.semantic.ocr import EnsembleOCR
from streams.semantic.validator import compute_semantic_score
from fusion.meta_classifier import SynthDocMetaClassifier
from training.train_spatial import DocumentDataset


def extract_features(data_dir: str, split: str, device: torch.device) -> tuple:
    """Extract feature vectors from all three streams for a given split.

    Returns:
        X: Feature matrix (N, 9)
        y: Labels (N,)
    """
    # Load models
    spatial_model = SpatialForensicsStream(pretrained=False).to(device)
    try:
        ckpt = torch.load("checkpoints/spatial_best.pth", map_location=device, weights_only=False)
        spatial_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    except FileNotFoundError:
        print("WARNING: spatial_best.pth not found")
    spatial_model.eval()

    freq_model = FrequencyForensicsStream(pretrained=False).to(device)
    try:
        ckpt = torch.load("checkpoints/frequency_best.pth", map_location=device, weights_only=False)
        freq_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    except FileNotFoundError:
        print("WARNING: frequency_best.pth not found")
    freq_model.eval()

    ocr_engine = EnsembleOCR(use_paddle=False)

    # Process images
    data_path = Path(data_dir)
    features_list = []
    labels_list = []

    from torchvision import transforms
    spatial_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for label_name, label_val in [("genuine", 0), ("synthetic", 1)]:
        label_dir = data_path / split / label_name
        if not label_dir.exists():
            continue

        for img_path in sorted(label_dir.glob("*")):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            try:
                pil_img = Image.open(str(img_path)).convert("RGB")
                img_np = np.array(pil_img)

                # Spatial
                spatial_input = spatial_transform(pil_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    spatial_out = spatial_model(spatial_input)
                spatial_score = float(spatial_out["spatial_score"].cpu().item())

                # Frequency
                freq_input = prepare_freq_tensor(img_np, size=224).unsqueeze(0).to(device)
                with torch.no_grad():
                    freq_out = freq_model(freq_input)
                freq_score = float(freq_out["frequency_score"].cpu().item())

                # Semantic
                ocr_result = ocr_engine.extract(pil_img)
                merged_text = ocr_result.get("merged", "")
                fields = ocr_result.get("fields", {})
                semantic_score = compute_semantic_score(
                    ocr_text=merged_text,
                    fields_found=len(fields),
                )

                w, h = pil_img.size
                file_size = img_path.stat().st_size

                feature_vec = [
                    spatial_score,
                    freq_score,
                    semantic_score,
                    0.85,  # spatial_conf
                    0.80,  # frequency_conf
                    0.75,  # semantic_conf
                    0,     # doc_type (default PAN)
                    min(w * h / (4000 * 3000), 1.0),
                    min(file_size / (10 * 1024 * 1024), 1.0),
                ]

                features_list.append(feature_vec)
                labels_list.append(label_val)

            except Exception as e:
                print(f"Skipping {img_path}: {e}")
                continue

    return np.array(features_list), np.array(labels_list)


def main():
    parser = argparse.ArgumentParser(description="Train Meta-Classifier Fusion")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("Extracting training features...")
    X_train, y_train = extract_features(args.data_dir, "train", device)
    print(f"Training set: {X_train.shape[0]} samples")

    print("Extracting validation features...")
    X_val, y_val = extract_features(args.data_dir, "val", device)
    print(f"Validation set: {X_val.shape[0]} samples")

    print("Training meta-classifier...")
    meta = SynthDocMetaClassifier()
    meta.train(X_train, y_train, X_cal=X_val, y_cal=y_val)

    save_path = "checkpoints/fusion_latest.joblib"
    os.makedirs("checkpoints", exist_ok=True)
    meta.save(save_path)
    print(f"Meta-classifier saved to {save_path}")

    # Quick evaluation
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    preds = []
    for i in range(X_val.shape[0]):
        feature_dict = {k: X_val[i, j] for j, k in enumerate([
            'spatial_score', 'frequency_score', 'semantic_score',
            'spatial_conf', 'frequency_conf', 'semantic_conf',
            'doc_type', 'resolution_norm', 'file_size_norm',
        ])}
        result = meta.predict(feature_dict)
        preds.append(result["fraud_probability"])

    preds = np.array(preds)
    binary_preds = (preds > 0.5).astype(int)

    print(f"\nValidation Results:")
    print(f"  Accuracy:  {accuracy_score(y_val, binary_preds):.4f}")
    print(f"  F1 Score:  {f1_score(y_val, binary_preds):.4f}")
    print(f"  AUC-ROC:   {roc_auc_score(y_val, preds):.4f}")


if __name__ == "__main__":
    main()
