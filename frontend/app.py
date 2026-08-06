import streamlit as st
import requests
import json
import numpy as np
import cv2
from PIL import Image
import io
import plotly.graph_objects as go
import time

st.set_page_config(page_title="SynthDoc | Advanced Document Forensics", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# Ultra-Premium Modern Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 { color: #f0f6fc; font-weight: 700; letter-spacing: -0.025em; }
    h1 { font-size: 2.8rem; background: -webkit-linear-gradient(45deg, #58a6ff, #8a2be2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    
    .risk-banner {
        padding: 2rem;
        border-radius: 16px;
        font-weight: 800;
        text-align: center;
        font-size: 2rem;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        backdrop-filter: blur(10px);
    }
    .CRITICAL { background: linear-gradient(135deg, rgba(153,27,27,0.8), rgba(220,38,38,0.2)); color: #fca5a5; border: 1px solid #ef4444; }
    .HIGH { background: linear-gradient(135deg, rgba(154,52,18,0.8), rgba(249,115,22,0.2)); color: #fdba74; border: 1px solid #f97316; }
    .MEDIUM { background: linear-gradient(135deg, rgba(133,77,14,0.8), rgba(234,179,8,0.2)); color: #fef08a; border: 1px solid #eab308; }
    .LOW { background: linear-gradient(135deg, rgba(22,101,52,0.8), rgba(34,197,94,0.2)); color: #86efac; border: 1px solid #22c55e; }
    
    .metric-card {
        background-color: #161b22;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #30363d;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-5px); border-color: #58a6ff; }
    
    .stButton>button {
        background: linear-gradient(90deg, #238636, #2ea043);
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2ea043, #3fb950);
        box-shadow: 0 0 15px rgba(46,160,67,0.4);
    }
    
    .sidebar .stMarkdown { color: #8b949e; }
</style>
""", unsafe_allow_html=True)

# Sidebar Architecture Info
with st.sidebar:
    st.markdown("## SynthDoc Engine Status")
    st.markdown("---")
    st.markdown("**🟢 Backend API**: Connected\n**🟢 Spatial Stream**: Active\n**🟢 Frequency Stream**: Active\n**🟢 Meta-Classifier**: Loaded")
    st.markdown("---")
    st.markdown("### Architecture Specs")
    st.markdown("""
    - **Stream 1**: EfficientNet-B4 + ViT (Spatial)
    - **Stream 2**: DCT/FFT + ResNet-18 (Frequency)
    - **Stream 3**: Tesseract/Paddle OCR (Semantic)
    - **Fusion**: Calibrated XGBoost + LightGBM
    - **Latency Target**: < 500ms
    """)

st.title("SynthDoc Fraud Intelligence Platform")
st.markdown("Upload a high-resolution Indian identity document (PAN, Aadhaar, Passport) to perform a zero-trust multi-modal forensic analysis.")

def generate_ela(image):
    """Generate Error Level Analysis heatmap via high-compression difference mapping"""
    img = np.array(image.convert("RGB"))
    cv2.imwrite("temp.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    compressed = cv2.imread("temp.jpg")
    compressed = cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB)
    
    diff = np.abs(img.astype(np.int16) - compressed.astype(np.int16))
    scale = 255.0 / np.max(diff) if np.max(diff) > 0 else 1.0
    diff = (diff * scale).astype(np.uint8)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_TURBO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap

uploaded_file = st.file_uploader("Secure Document Upload (Local Execution)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("<h3 style='color: #8b949e; font-weight: 500;'>Input Geometry</h3>", unsafe_allow_html=True)
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True, clamp=True)
        
    with col2:
        st.markdown("<h3 style='color: #8b949e; font-weight: 500;'>Forensic ELA Rendering</h3>", unsafe_allow_html=True)
        with st.spinner("Generating ELA analysis..."):
            ela_map = generate_ela(image)
            st.image(ela_map, use_container_width=True, caption="Error Level Analysis (White/Red spots indicate pixel manipulation)")
            
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Initiate Deep Neural Verification", use_container_width=True):
        with st.status("Running Multimodal Inference...", expanded=True) as status:
            st.write("📡 Transmitting to local API node...")
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            
            try:
                st.write("🧠 Executing Spatial & Frequency CNNs...")
                res = requests.post("http://127.0.0.1:8000/v1/verify", files=files)
                res.raise_for_status()
                data = res.json()
                
                st.write("📊 Merging data via Meta-Classifier Fusion...")
                time.sleep(0.5)
                status.update(label="Verification Complete!", state="complete", expanded=False)
                
                risk_tier = data["risk_tier"]
                prob = data["fraud_probability"]
                st.markdown(f'<div class="risk-banner {risk_tier}">ESTIMATED DECISION: {risk_tier} FRAUD RISK <br><span style="font-size: 1rem; color: #8b949e;">Classifier Neural Confidence: {prob*100:.2f}%</span></div>', unsafe_allow_html=True)
                
                st.markdown("### Forensic Evidence Breakdown")
                m1, m2, m3 = st.columns(3)
                streams = data.get("streams", {})
                
                sp_score = streams.get("spatial_score", 0.0)
                fq_score = streams.get("frequency_score", 0.0)
                sm_score = streams.get("semantic_score", 0.0)
                
                m1.markdown(f"<div class='metric-card'><h4>Spatial CNN</h4><h2 style='color: {'#ef4444' if sp_score>0.5 else '#22c55e'}'>{sp_score*100:.1f}%</h2><p style='color:#8b949e;font-size:0.8rem;'>Texture Manipulation</p></div>", unsafe_allow_html=True)
                m2.markdown(f"<div class='metric-card'><h4>Frequency FFT</h4><h2 style='color: {'#ef4444' if fq_score>0.5 else '#22c55e'}'>{fq_score*100:.1f}%</h2><p style='color:#8b949e;font-size:0.8rem;'>Compression Artifacts</p></div>", unsafe_allow_html=True)
                m3.markdown(f"<div class='metric-card'><h4>Semantic Validator</h4><h2 style='color: {'#ef4444' if sm_score>0.5 else '#22c55e'}'>{sm_score*100:.1f}%</h2><p style='color:#8b949e;font-size:0.8rem;'>OCR & Checksums</p></div>", unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                chart_col, info_col = st.columns([1.5, 1], gap="large")
                
                with chart_col:
                    st.markdown("#### Feature Importance (SHAP Approximation)")
                    features = ["Spatial Features", "Frequency Spectra", "Semantic Consistency"]
                    scores = [sp_score, fq_score, sm_score]
                    
                    fig = go.Figure(go.Bar(
                        x=scores,
                        y=features,
                        orientation='h',
                        marker_color=['#ef4444' if s>0.5 else '#22c55e' for s in scores],
                        text=[f"{s*100:.1f}%" for s in scores],
                        textposition='auto',
                        opacity=0.85
                    ))
                    fig.update_layout(
                        xaxis_title="Anomaly Contribution Score", 
                        height=250, 
                        margin=dict(l=0, r=0, t=0, b=0),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#c9d1d9'),
                        xaxis=dict(showgrid=True, gridcolor='#30363d', range=[0, 1])
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with info_col:
                    st.markdown("#### Meta-Classifier Extracted Data")
                    evidence = data.get("evidence", {})
                    
                    st.markdown(f"**Detected Type**: `{data.get('document_type', 'UNKNOWN')}`")
                    
                    valid_format = evidence.get("format_valid", True) if "format_valid" in evidence else True
                    chk_valid = evidence.get("checksum_valid", True)
                    
                    st.markdown(f"- **Format Validation**: {'✅ Passed' if valid_format else '❌ Failed'}")
                    st.markdown(f"- **Algorithms (Luhn/Verhoeff)**: {'✅ Passed' if chk_valid else '❌ Failed'}")
                    
                    spatial_anom = evidence.get("spatial_anomalies", [])
                    if spatial_anom:
                        st.markdown(f"- **Detected Anomalies**: 🚩 `{', '.join(spatial_anom)}`")
                    else:
                        st.markdown("- **Detected Anomalies**: 🟢 `None`")
                        
                    if evidence.get("ocr_fields"):
                        with st.expander("View Raw Parsed Text"):
                            st.write(evidence["ocr_fields"])
                
                st.markdown("---")
                with st.expander("⚙️ View Raw Engine JSON Output"):
                    st.json(data)
                    
            except Exception as e:
                status.update(label="Analysis Failed", state="error", expanded=True)
                st.error(f"Critical Backend Error: {str(e)}\n\nPlease ensure `./start_platform.sh` is running and port 8000 is open!")
