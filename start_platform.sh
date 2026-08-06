#!/bin/bash
echo "🚀 Starting SynthDoc AI Platform Locally..."
echo "==================================="

# 1. Start the FastAPI Backend in the background
echo "[1/2] Launching Neural Verification API (Port 8000)..."
uvicorn api.main:app --host 127.0.0.1 --port 8000 > api_server.log 2>&1 &
API_PID=$!

sleep 2

# 2. Start Streamlit in the foreground (so user can see logs)
echo "[2/2] Launching Premium Dashboard (Port 8501)..."
streamlit run frontend/app.py --server.port 8501

# Cleanup trap
trap "kill $API_PID; exit" INT TERM
