"""SynthDoc FastAPI application.

Provides the /v1/verify endpoint for multi-modal document forensics.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import traceback

from api.pipeline import run_pipeline, _load_models

app = FastAPI(
    title="SynthDoc Forensic API",
    description="Multi-modal AI engine for detecting synthetic Indian identity documents.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Pre-load models on startup for faster first inference."""
    try:
        _load_models()
        print("[SynthDoc] All models loaded successfully.")
    except Exception as e:
        print(f"[SynthDoc] Warning: Model pre-loading failed: {e}")
        print("[SynthDoc] Models will be lazy-loaded on first request.")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "engine": "SynthDoc v1.0"}


@app.post("/v1/verify")
async def verify_document(file: UploadFile = File(...)):
    """Verify a document image across all three forensic streams.

    Args:
        file: Uploaded image file (JPEG/PNG).

    Returns:
        JSON with fraud_probability, risk_tier, stream scores, and evidence.
    """
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG or PNG.",
        )

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        result = await run_pipeline(image, image_bytes, file.filename)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
