#!/usr/bin/env bash
# ────────────────────────────────────────────────
# SynthDoc Vertex AI Training Launcher
# ────────────────────────────────────────────────
# Prerequisites:
#   1. gcloud CLI installed and authenticated
#   2. Docker installed
#   3. A GCS bucket for data & artifacts
#
# Usage:
#   ./launch_vertex_training.sh <PROJECT_ID> <GCS_BUCKET> [REGION]
#
# Example:
#   ./launch_vertex_training.sh my-project-123 gs://synthdoc-training us-central1
# ────────────────────────────────────────────────

set -euo pipefail

PROJECT_ID="${1:?Usage: $0 <PROJECT_ID> <GCS_BUCKET> [REGION]}"
GCS_BUCKET="${2:?Usage: $0 <PROJECT_ID> <GCS_BUCKET> [REGION]}"
REGION="${3:-us-central1}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/synthdoc/trainer:${TIMESTAMP}"

echo "╔══════════════════════════════════════════════╗"
echo "║      SynthDoc Vertex AI Training Setup       ║"
echo "╠══════════════════════════════════════════════╣"
echo "║ Project:  ${PROJECT_ID}"
echo "║ Bucket:   ${GCS_BUCKET}"
echo "║ Region:   ${REGION}"
echo "║ Image:    ${IMAGE_URI}"
echo "╚══════════════════════════════════════════════╝"

# ── Step 1: Upload dataset to GCS ──────────────────
echo ""
echo "📦 Step 1: Uploading dataset to GCS..."
if gsutil ls "${GCS_BUCKET}/data/raw/train/genuine/" &>/dev/null; then
    echo "   Dataset already exists on GCS, skipping upload."
else
    gsutil -m cp -r data/raw/* "${GCS_BUCKET}/data/raw/"
    echo "   ✅ Dataset uploaded to ${GCS_BUCKET}/data/raw/"
fi

# ── Step 2: Create Artifact Registry repo ──────────
echo ""
echo "🏗️  Step 2: Setting up Artifact Registry..."
gcloud artifacts repositories create synthdoc \
    --repository-format=docker \
    --location="${REGION}" \
    --project="${PROJECT_ID}" 2>/dev/null || echo "   Repository already exists."

gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# ── Step 3: Build & push container ─────────────────
echo ""
echo "🐳 Step 3: Building and pushing training container..."
docker build -f infra/docker/Dockerfile.vertex -t "${IMAGE_URI}" .
docker push "${IMAGE_URI}"
echo "   ✅ Container pushed to ${IMAGE_URI}"

# ── Step 4: Submit Spatial Training Job ────────────
echo ""
echo "🚀 Step 4: Submitting SPATIAL stream training job..."
SPATIAL_JOB="synthdoc-spatial-${TIMESTAMP}"

gcloud ai custom-jobs create \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --display-name="${SPATIAL_JOB}" \
    --worker-pool-spec="\
machine-type=g2-standard-8,\
accelerator-type=NVIDIA_L4,\
accelerator-count=1,\
replica-count=1,\
container-image-uri=${IMAGE_URI}" \
    --args="--stream=spatial,--epochs=50,--batch_size=32,--lr=1e-4,--data_root=/gcs/${GCS_BUCKET#gs://}/data/raw,--checkpoint_dir=/gcs/${GCS_BUCKET#gs://}/checkpoints"

echo "   ✅ Spatial job submitted: ${SPATIAL_JOB}"

# ── Step 5: Submit Frequency Training Job ──────────
echo ""
echo "🚀 Step 5: Submitting FREQUENCY stream training job..."
FREQ_JOB="synthdoc-frequency-${TIMESTAMP}"

gcloud ai custom-jobs create \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --display-name="${FREQ_JOB}" \
    --worker-pool-spec="\
machine-type=g2-standard-8,\
accelerator-type=NVIDIA_L4,\
accelerator-count=1,\
replica-count=1,\
container-image-uri=${IMAGE_URI}" \
    --args="--stream=frequency,--epochs=40,--batch_size=32,--lr=1e-4,--data_root=/gcs/${GCS_BUCKET#gs://}/data/raw,--checkpoint_dir=/gcs/${GCS_BUCKET#gs://}/checkpoints"

echo "   ✅ Frequency job submitted: ${FREQ_JOB}"

echo ""
echo "═══════════════════════════════════════════════"
echo "Both training jobs submitted!"
echo ""
echo "Monitor progress:"
echo "  gcloud ai custom-jobs list --project=${PROJECT_ID} --region=${REGION}"
echo ""
echo "View logs:"
echo "  gcloud ai custom-jobs stream-logs ${SPATIAL_JOB} --project=${PROJECT_ID} --region=${REGION}"
echo "  gcloud ai custom-jobs stream-logs ${FREQ_JOB} --project=${PROJECT_ID} --region=${REGION}"
echo ""
echo "After training completes, download checkpoints:"
echo "  gsutil cp ${GCS_BUCKET}/checkpoints/*.pth checkpoints/"
echo "═══════════════════════════════════════════════"


