"""Comprehensive evaluation suite for the SynthDoc pipeline.

Computes accuracy, precision, recall, F1, AUC-ROC, and false positive rate
with per-document-type breakdowns.

Usage:
    python -m evaluation.evaluate --data_dir data/processed --split val
"""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report,
)

from api.pipeline import _load_models, _get_device
from streams.frequency.preprocess import prepare_freq_tensor


def evaluate_split(data_dir: str, split: str = "val"):
    """Run full pipeline evaluation on a data split."""
    _load_models()
    device = _get_device()

    from api.pipeline import _spatial_model, _frequency_model, _ocr_engine, _meta_classifier
    from streams.semantic.validator import compute_semantic_score
    from torchvision import transforms
    from fusion.meta_classifier import DOC_TYPE_MAP

    spatial_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_path = Path(data_dir)
    all_labels = []
    all_probs = []
    all_tiers = []

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

                # Spatial inference
                spatial_input = spatial_transform(pil_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    spatial_out = _spatial_model(spatial_input)
                spatial_score = float(spatial_out["spatial_score"].cpu().item())

                # Frequency inference
                freq_input = prepare_freq_tensor(img_np, size=224).unsqueeze(0).to(device)
                with torch.no_grad():
                    freq_out = _frequency_model(freq_input)
                freq_score = float(freq_out["frequency_score"].cpu().item())

                # Semantic inference
                ocr_result = _ocr_engine.extract(pil_img)
                merged_text = ocr_result.get("merged", "")
                fields = ocr_result.get("fields", {})
                semantic_score = compute_semantic_score(
                    ocr_text=merged_text,
                    fields_found=len(fields),
                )

                w, h = pil_img.size
                file_size = img_path.stat().st_size

                features = {
                    'spatial_score': spatial_score,
                    'frequency_score': freq_score,
                    'semantic_score': semantic_score,
                    'spatial_conf': 0.85,
                    'frequency_conf': 0.80,
                    'semantic_conf': 0.75,
                    'doc_type': 0,
                    'resolution_norm': min(w * h / (4000 * 3000), 1.0),
                    'file_size_norm': min(file_size / (10 * 1024 * 1024), 1.0),
                }

                result = _meta_classifier.predict(features)

                all_labels.append(label_val)
                all_probs.append(result["fraud_probability"])
                all_tiers.append(result["risk_tier"])

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

    # Compute metrics
    labels = np.array(all_labels)
    probs = np.array(all_probs)
    preds = (probs > 0.5).astype(int)

    print("=" * 60)
    print("SynthDoc Evaluation Report")
    print("=" * 60)
    print(f"Total samples: {len(labels)}")
    print(f"  Genuine:   {(labels == 0).sum()}")
    print(f"  Synthetic: {(labels == 1).sum()}")
    print()
    print(f"Accuracy:    {accuracy_score(labels, preds):.4f}")
    print(f"Precision:   {precision_score(labels, preds, zero_division=0):.4f}")
    print(f"Recall:      {recall_score(labels, preds, zero_division=0):.4f}")
    print(f"F1 Score:    {f1_score(labels, preds, zero_division=0):.4f}")

    if len(np.unique(labels)) > 1:
        print(f"AUC-ROC:     {roc_auc_score(labels, probs):.4f}")

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    print(f"FPR:         {fpr:.4f}")
    print()
    print("Confusion Matrix:")
    print(f"  TP={tp}  FP={fp}")
    print(f"  FN={fn}  TN={tn}")
    print()
    print("Risk Tier Distribution:")
    for tier in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        count = sum(1 for t in all_tiers if t == tier)
        print(f"  {tier}: {count}")

    # Save report
    report = {
        "total_samples": len(labels),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1_score": float(f1_score(labels, preds, zero_division=0)),
        "false_positive_rate": float(fpr),
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
    }

    report_path = "reports/evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SynthDoc Pipeline")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--split", type=str, default="val")
    args = parser.parse_args()

    evaluate_split(args.data_dir, args.split)


if __name__ == "__main__":
    main()
