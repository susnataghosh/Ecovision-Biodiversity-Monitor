import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from PIL import Image, ImageOps
import tf_keras as keras
from pathlib import Path
import requests
import io
import re
st.set_page_config(page_title="EcoVision - Indian Wildlife AI", page_icon="🐾", layout="wide")

@st.cache_resource
def load_tm_model():
    model_path = Path(__file__).parent / "keras_model.h5"
    labels_path = Path(__file__).parent / "labels.txt"
    model = keras.models.load_model(str(model_path), compile=False)
    with open(labels_path, "r") as f:
        class_names = [line.strip().split(" ", 1)[1] for line in f.readlines() if line.strip()]
    return model, class_names

if "live_logs" not in st.session_state:
    _, labels = load_tm_model()
    np.random.seed(42)
    cameras = ['Zone A (River Bank)', 'Zone B (Deep Jungle)', 'Canopy Grid 4']
    dates = [datetime(2026, 11, 1) + timedelta(days=int(np.random.randint(0, 14)), hours=int(np.random.randint(0, 24))) for _ in range(25)]
    st.session_state["live_logs"] = pd.DataFrame({
        'Timestamp': [d.strftime("%Y-%m-%d %H:%M") for d in dates],
        'Camera Deployment': np.random.choice(cameras, size=25),
        'Species Detected': np.random.choice(labels, size=25),
        'Confidence (%)': np.round(np.random.uniform(78.0, 99.5, size=25), 1),
        'Hour': [d.hour for d in dates],
        'Source': ['Baseline Field Log'] * 25
    })
st.title("🐾 EcoVision: Biodiversity Monitor")
st.caption("EVS Project Solution — Automated Species Classification Pipeline")

tab1, tab2 = st.tabs(["📸 AI Wildlife Classifier", "📊 Field Analytics Dashboard"])

with tab1:
    st.markdown("### Upload Image")
    
    model_file = Path(__file__).parent / "keras_model.h5"
    labels_file = Path(__file__).parent / "labels.txt"
    
    if not model_file.exists() or not labels_file.exists():
        st.error("❌ Move `keras_model.h5` and `labels.txt` out of `converted_keras` directly into your `EVS` folder!")
    else:
        image = None
        input_mode = st.radio(
            "Choose input method:",
            ["📁 Upload File", "📷 Live Camera", "🔗 Direct Image URL"],
            horizontal=True
        )

        if input_mode == "📁 Upload File":
            uploaded_file = st.file_uploader("Camera Trap / Field Image Input", type=["jpg", "jpeg", "png", "webp"])
            if uploaded_file:
                image = Image.open(uploaded_file).convert("RGB")

        elif input_mode == "📷 Live Camera":
            cam_file = st.camera_input("Snap a photo of a wildlife image or screen")
            if cam_file:
                image = Image.open(cam_file).convert("RGB")

        else:
            url_input = st.text_input("Paste Google Drive share link or direct image URL:")
            if url_input:
                try:
                    drive_match = re.search(r"/d/([a-zA-Z0-9_-]+)|id=([a-zA-Z0-9_-]+)", url_input)
                    if "://google.com" in url_input and drive_match:
                        file_id = drive_match.group(1) or drive_match.group(2)
                        fetch_url = f"https://googleusercontent.com{file_id}"
                    else:
                        fetch_url = url_input

                    headers = {"User-Agent": "Mozilla/5.0"}
                    response = requests.get(fetch_url, headers=headers, timeout=10)
                    response.raise_for_status()
                    image = Image.open(io.BytesIO(response.content)).convert("RGB")
                except Exception as e:
                    st.error(f"❌ Could not load image ({e}). Try downloading the photo to your phone first and using 'Upload File'.")
        if image is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Uploaded Frame", use_container_width=True)   
            with col2:
                if st.button("🔎 Classify Species", type="primary"):
                    with st.spinner("Running neural network inference..."):
                        model, class_names = load_tm_model()
                        
                        size = (224, 224)
                        image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
                        image_array = np.asarray(image_resized, dtype=np.float32)
                        normalized_image_array = (image_array / 127.5) - 1
                        data = np.expand_dims(normalized_image_array, axis=0)
                        
                        prediction = model.predict(data)[0]
                        index = int(np.argmax(prediction))
                        species = class_names[index]
                        confidence = float(prediction[index])
                        assigned_cam = np.random.choice(['Zone A (River Bank)', 'Zone B (Deep Jungle)', 'Canopy Grid 4'])
                        # Log live detection to Tab 2 dashboard
                        now = datetime.now()
                        logged_label = "Unidentified Species" if confidence < 0.65 and "empty" not in species.lower() and "background" not in species.lower() else species
                        new_entry = pd.DataFrame([{
                            'Timestamp': now.strftime("%Y-%m-%d %H:%M:%S"),
                            'Camera Deployment': assigned_cam,
                            'Species Detected': logged_label,
                            'Confidence (%)': round(confidence * 100, 1),
                            'Hour': now.hour,
                            'Source': '🔴 Live Demo Upload'
                        }])
                        st.session_state["live_logs"] = pd.concat([new_entry, st.session_state["live_logs"]], ignore_index=True)
                        st.markdown("### **Classification Results**")
                        if "blank" in species.lower():
                            st.error(f"**False Trigger Filtered:** {species} ({confidence*100:.1f}% Confidence)")
                        elif confidence < 0.75:
                            st.warning(f"⚠️ **Unidentified Species** (Highest match: {species} at {confidence*100:.1f}%)")
                        else:
                            st.success(f"**Detected Species:** {species}")
                            st.metric("Model Confidence", f"{confidence*100:.1f}%", f"Station: {assigned_cam}")
                        
                        st.markdown("#### Confidence Breakdown")
                        for i, name in enumerate(class_names):
                            st.progress(float(prediction[i]), text=f"{name}: {prediction[i]*100:.1f}%")

with tab2:
    st.markdown("### Real-Time Field Telemetry & Detection Log")
    
    df_all = st.session_state["live_logs"]
    live_only = df_all[df_all['Source'] == '🔴 Live Demo Upload']
    
    # Top summary metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Logged Detections", len(df_all))
    m2.metric("Live Session Uploads", len(live_only))
    m3.metric("Unique Species Found", df_all['Species Detected'].nunique())
    
    filter_choice = st.radio(
        "Select Data View:",
        ["📊 Combined Telemetry (Baseline + Live Uploads)", "🔴 Only My Live Uploads"],
        horizontal=True
    )
    
    df_display = live_only if filter_choice == "🔴 Only My Live Uploads" else df_all
    
    if len(df_display) == 0:
        st.info("No live images classified yet! Go to Tab 1, upload a photo, and click 'Classify Species' to see it appear here.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                px.pie(df_display, names='Species Detected', title="Species Abundance Breakdown"),
                use_container_width=True
            )
        with col2:
            st.plotly_chart(
                px.histogram(
                    df_display,
                    x='Hour',
                    color='Species Detected',
                    nbins=24,
                    barmode='group',
                    title="Activity Patterns by Hour (24hr)"
                ),
                use_container_width=True
            )
        
        st.markdown("#### Recent Detection Logs")
        st.dataframe(df_display[['Timestamp', 'Species Detected', 'Confidence (%)', 'Camera Deployment', 'Source']], use_container_width=True)
