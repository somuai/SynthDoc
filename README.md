# SynthDoc: Multi-Modal AI Document Forensics Framework

SynthDoc is a multi-modal AI framework designed to detect forged, tampered, and synthetic identity documents (PAN, Aadhaar, Passport, Voter ID, Driving License, UPI QR). It utilizes a three-stream architecture fused through a calibrated meta-classifier to output risk probabilities and evidence logs.

🔗 **Live Explainer & Demo**: [https://explainer-tau.vercel.app/](https://explainer-tau.vercel.app/)

---

## 🎬 Platform Demonstration

![SynthDoc Live Demo](assets/demo.gif)

---

## 🏛️ System Architecture

```
                                 ┌────────────────────────┐
                                 │  Input Document Image  │
                                 └───────────┬────────────┘
                                             │
               ┌─────────────────────────────┼─────────────────────────────┐
               ▼                             ▼                             ▼
   ┌───────────────────────┐     ┌───────────────────────┐     ┌───────────────────────┐
   │    Spatial Stream     │     │   Frequency Stream    │     │    Semantic Stream    │
   │ EfficientNet-B4 + ViT │     │  ResNet-18 (FFT/DCT)  │     │   Ensemble OCR +      │
   │   (Cross-Attention)   │     │ (Spectral Anomalies)  │     │ Luhn/Verhoeff Checks  │
   └───────────┬───────────┘     └───────────┬───────────┘     └───────────┬───────────┘
               │                             │                             │
               └─────────────────────────────┼─────────────────────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │  Meta-Classifier Fusion   │
                               │ XGBoost + LightGBM + Iso  │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │  Calibrated Fraud Score   │
                               │   & Forensic Dashboard    │
                               └───────────────────────────┘
```

---

## 🌟 Key Features

1. **Spatial Stream (CNN + ViT)**: Combines EfficientNet-B4 local feature maps with Vision Transformer (ViT) global attention vectors via a Cross-Attention block to identify localized pixel manipulations.
2. **Frequency Stream (FFT + DCT)**: Analyzes 2-channel Fourier (FFT) and Discrete Cosine Transform (DCT) magnitude spectra to detect resampling, double-JPEG compression, and generative synthesis artifacts.
3. **Semantic Stream (OCR + Algorithmic Verification)**: Performs ensemble text extraction (Tesseract/PaddleOCR) and cross-validates document numbers against Luhn (PAN) and Verhoeff (Aadhaar) checksum algorithms.
4. **Calibrated Fusion**: Fuses stream outputs using an ensemble of XGBoost and LightGBM models with Isotonic Regression for calibrated risk probabilities.
5. **Interactive Forensic Dashboard**: Streamlit-based web interface featuring real-time ELA (Error Level Analysis) heatmaps, SHAP feature importance breakdown, and raw evidence logs.

---

## 🔗 Live Interactive Links

- **Explainer & App Web Portal**: [https://explainer-tau.vercel.app/](https://explainer-tau.vercel.app/)

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- PyTorch 2.1+

### 2. Installation
```bash
git clone https://github.com/somuai/SynthDoc.git
cd SynthDoc
pip install -r requirements.txt
```

### 3. Launching Platform
Run the unified startup script to launch both the FastAPI backend (`:8000`) and the Streamlit dashboard (`:8501`):

```bash
./start_platform.sh
```

Navigate to `http://localhost:8501` in your browser.

---

## 📂 Project Structure

```
SynthDoc/
├── assets/
│   └── demo.gif              # Visual demo animation
├── api/
│   ├── main.py              # FastAPI server (/v1/verify)
│   └── pipeline.py           # Multi-stream async pipeline orchestrator
├── streams/
│   ├── spatial/             # CNN-ViT cross-attention model
│   ├── frequency/           # ResNet-18 FFT/DCT spectral analyzer
│   └── semantic/            # Ensemble OCR & Luhn/Verhoeff checksums
├── fusion/
│   └── meta_classifier.py    # Calibrated XGBoost + LightGBM fusion
├── frontend/
│   └── app.py                # Streamlit dark-mode dashboard
├── training/
│   ├── train_spatial.py      # Spatial stream training script
│   ├── train_frequency.py    # Frequency stream training script
│   └── train_fusion.py       # Meta-classifier training script
├── evaluation/
│   └── evaluate.py           # Comprehensive evaluation suite
├── tests/                    # Pytest unit & integration test suite
├── configs/
│   └── model_meta.json       # Model architecture metadata
└── start_platform.sh         # One-command platform launcher
```

---

## 🧪 Running Tests

To run the automated unit and integration tests:

```bash
pytest tests/
```

---

## 📄 License

MIT License.
