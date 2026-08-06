"""SynthDoc forensic analysis pipeline.

Orchestrates the three-stream architecture:
1. Spatial: CNN+ViT cross-attention for pixel manipulation detection
2. Frequency: DCT/FFT spectral analysis for compression artifacts
3. Semantic: OCR + checksum validation for field integrity

All streams run concurrently and feed into a meta-classifier fusion.
"""

import asyncio
import re
import numpy as np
import torch
from PIL import Image
from fusion.meta_classifier import SynthDocMetaClassifier

# Global model instances (lazy-loaded)
_spatial_model = None
_frequency_model = None
_ocr_engine = None
_meta_classifier = None

# Document type numeric mapping for feature vector
DOC_TYPE_MAP = {
    'PAN_CARD': 0, 'AADHAAR': 1, 'PASSPORT': 2,
    'VOTER_ID': 3, 'DRIVING_LICENSE': 4, 'UPI_QR': 5,
}


def _get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Disabled MPS due to known macOS PyTorch segfaults with >1GB model checkpoints
    # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #     return torch.device("mps")
    return torch.device("cpu")


def _load_models():
    """Lazy-load all stream models."""
    global _spatial_model, _frequency_model, _ocr_engine, _meta_classifier

    device = _get_device()

    if _spatial_model is None:
        from streams.spatial.model import SpatialForensicsStream
        _spatial_model = SpatialForensicsStream(pretrained=False).to(device)
        try:
            ckpt = torch.load("checkpoints/spatial_best.pth", map_location=device, weights_only=False)
            _spatial_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        except FileNotFoundError:
            print("WARNING: spatial_best.pth not found, using randomly initialized weights!")
        _spatial_model.eval()

    if _frequency_model is None:
        from streams.frequency.model import FrequencyForensicsStream
        _frequency_model = FrequencyForensicsStream(pretrained=False).to(device)
        try:
            ckpt = torch.load("checkpoints/frequency_best.pth", map_location=device, weights_only=False)
            _frequency_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        except FileNotFoundError:
            print("WARNING: frequency_best.pth not found, using randomly initialized weights!")
        _frequency_model.eval()

    if _ocr_engine is None:
        from streams.semantic.ocr import EnsembleOCR
        _ocr_engine = EnsembleOCR(use_paddle=False)

    if _meta_classifier is None:
        _meta_classifier = SynthDocMetaClassifier()
        try:
            _meta_classifier.load("checkpoints/fusion_latest.joblib")
        except FileNotFoundError:
            print("WARNING: fusion_latest.joblib not found, using randomly initialized weights!")


def _classify_document_type(filename: str, ocr_text: str = "") -> str:
    """Heuristic document type classification based on filename and OCR text."""
    fname_lower = (filename or "").lower()
    text_lower = ocr_text.lower()

    if "pan" in fname_lower or "permanent account" in text_lower:
        return "PAN_CARD"
    elif "aadhaar" in fname_lower or "aadhaar" in text_lower or "uidai" in text_lower:
        return "AADHAAR"
    elif "passport" in fname_lower or "passport" in text_lower:
        return "PASSPORT"
    elif "voter" in fname_lower or "election" in text_lower:
        return "VOTER_ID"
    elif "driv" in fname_lower or "driving" in text_lower or "license" in text_lower:
        return "DRIVING_LICENSE"
    elif "qr" in fname_lower or "upi" in text_lower:
        return "UPI_QR"
    else:
        return "PAN_CARD"  # Default


async def _run_spatial(image: Image.Image) -> dict:
    """Run the spatial forensics stream."""
    device = _get_device()
    img_np = np.array(image.resize((224, 224)))
    img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        result = _spatial_model(img_tensor)

    return {
        "spatial_score": float(result["spatial_score"].cpu().item()),
        "spatial_confidence": 0.85,
    }


async def _run_frequency(image: Image.Image) -> dict:
    """Run the frequency forensics stream."""
    device = _get_device()
    img_np = np.array(image)

    from streams.frequency.preprocess import prepare_freq_tensor
    freq_tensor = prepare_freq_tensor(img_np, size=224).unsqueeze(0).to(device)

    with torch.no_grad():
        result = _frequency_model(freq_tensor)

    return {
        "frequency_score": float(result["frequency_score"].cpu().item()),
        "frequency_confidence": 0.80,
    }


async def _run_semantic(image: Image.Image, doc_type: str) -> dict:
    """Run the semantic forensics stream."""
    from streams.semantic.validator import (
        validate_pan, validate_aadhaar, validate_upi_vpa, compute_semantic_score
    )

    ocr_result = _ocr_engine.extract(image)
    merged_text = ocr_result.get("merged", "")

    # Run validators based on document type
    pan_result = None
    aadhaar_result = None
    vpa_result = None
    checksum_valid = None

    if doc_type == "PAN_CARD":
        pan_match = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]', merged_text.upper())
        if pan_match:
            pan_result = validate_pan(pan_match.group())
            checksum_valid = pan_result.get("checksum_valid")

    elif doc_type == "AADHAAR":
        aadhaar_match = re.search(r'\d{4}\s*\d{4}\s*\d{4}', merged_text)
        if aadhaar_match:
            aadhaar_num = aadhaar_match.group().replace(' ', '')
            aadhaar_result = validate_aadhaar(aadhaar_num)
            checksum_valid = aadhaar_result.get("checksum_valid")

    elif doc_type == "UPI_QR":
        vpa_match = re.search(r'[a-zA-Z0-9.]+@[a-zA-Z]+', merged_text)
        if vpa_match:
            vpa_result = validate_upi_vpa(vpa_match.group())
            checksum_valid = vpa_result.get("is_valid")

    fields = ocr_result.get("fields", {})
    semantic_score = compute_semantic_score(
        ocr_text=merged_text,
        checksum_valid=checksum_valid,
        fields_found=len(fields),
    )

    return {
        "semantic_score": semantic_score,
        "semantic_confidence": 0.75,
        "checksum_valid": checksum_valid,
        "ocr_fields": fields,
    }


async def run_pipeline(
    image: Image.Image,
    image_bytes: bytes,
    filename: str = None,
) -> dict:
    """Run the full SynthDoc verification pipeline.

    Orchestrates all three streams concurrently, feeds results into
    the meta-classifier fusion engine, and returns calibrated risk assessment.
    """
    _load_models()

    width, height = image.size
    doc_type = _classify_document_type(filename or "")

    # Run all three streams concurrently
    spatial_result, freq_result, semantic_result = await asyncio.gather(
        _run_spatial(image),
        _run_frequency(image),
        _run_semantic(image, doc_type),
    )

    # Prepare feature vector for meta-classifier
    features = {
        'spatial_score': spatial_result['spatial_score'],
        'frequency_score': freq_result['frequency_score'],
        'semantic_score': semantic_result['semantic_score'],
        'spatial_conf': spatial_result['spatial_confidence'],
        'frequency_conf': freq_result['frequency_confidence'],
        'semantic_conf': semantic_result['semantic_confidence'],
        'doc_type': DOC_TYPE_MAP.get(doc_type, 0),
        'resolution_norm': min(width * height / (4000 * 3000), 1.0),
        'file_size_norm': min(len(image_bytes) / (10 * 1024 * 1024), 1.0),
    }

    # Meta-classifier fusion
    fusion_result = _meta_classifier.predict(features)

    # Determine spatial anomalies from heatmap analysis
    spatial_anomalies = []
    if spatial_result['spatial_score'] > 0.6:
        spatial_anomalies.append("texture_inconsistency")
    if spatial_result['spatial_score'] > 0.8:
        spatial_anomalies.append("possible_gan_artifact")

    return {
        "fraud_probability": fusion_result["fraud_probability"],
        "risk_tier": fusion_result["risk_tier"],
        "document_type": doc_type,
        "streams": {
            "spatial_score": spatial_result["spatial_score"],
            "frequency_score": freq_result["frequency_score"],
            "semantic_score": semantic_result["semantic_score"],
        },
        "evidence": {
            "checksum_valid": semantic_result.get("checksum_valid"),
            "frequency_anomaly": freq_result["frequency_score"] > 0.6,
            "spatial_anomalies": spatial_anomalies,
            "ocr_fields": semantic_result.get("ocr_fields"),
        },
    }
