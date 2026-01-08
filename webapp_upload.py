# -*- coding: utf-8 -*-
"""
Cloud Version - Image Upload Face Detection
Created: Jan 08 2026
@author: KETSAR

Version untuk deployment cloud (Railway/Streamlit Cloud)
Menggunakan image upload karena WebRTC tidak support di cloud environment.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import requests
from pathlib import Path

# --- KONFIGURASI PAGE & STYLE ---
st.set_page_config(layout="wide", page_title="Face Detection - Cloud Version")

st.markdown("""
<style>
    .main { background-color: #0E1117; }
    h1 { color: #00A8E8; text-align: center; }
    h2, h3 { color: #FFFFFF; text-align: center; }
    .stMetric { background-color: #262730; border-radius: 10px; padding: 10px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# --- FUNGSI HELPER UNTUK MENGUNDUH MODEL ---
def download_file(url, file_path, file_name):
    """Mengunduh file dari URL dengan progress bar."""
    local_path = Path(file_path)
    if not local_path.exists():
        with st.spinner(f"Mengunduh model {file_name}..."):
            try:
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                progress_bar = st.progress(0)
                
                with open(local_path, "wb") as f:
                    downloaded_size = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            if total_size > 0:
                                progress = min(downloaded_size / total_size, 1.0)
                                progress_bar.progress(progress)
                
                progress_bar.empty()
                return True
            except Exception as e:
                st.error(f"Gagal mengunduh {file_name}: {e}")
                return False
    return True

# --- URL MODEL ---
MODEL_DIR = Path(__file__).parent / "model"
MODEL_DIR.mkdir(exist_ok=True)

MODELS_TO_DOWNLOAD = {
    "haarcascade_frontalface_default.xml": (
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
        MODEL_DIR / "haarcascade_frontalface_default.xml"
    ),
    "age_net.caffemodel": (
       "https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/age_net.caffemodel",
        MODEL_DIR / "age_net.caffemodel"
    ),
    "age_deploy.prototxt": (
        "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/age_deploy.prototxt",
        MODEL_DIR / "age_deploy.prototxt"
    ),
    "gender_net.caffemodel": (
        "https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/gender_net.caffemodel",
        MODEL_DIR / "gender_net.caffemodel"
    ),
    "gender_deploy.prototxt": (
        "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/gender_deploy.prototxt",
        MODEL_DIR / "gender_deploy.prototxt"
    ),
    "emotion-ferplus-8.onnx": (
        "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
        MODEL_DIR / "emotion-ferplus-8.onnx"
    ),
}

# Download semua model
with st.spinner("Memeriksa dan mengunduh model AI..."):
    for file_name, (url, path) in MODELS_TO_DOWNLOAD.items():
        download_file(url, path, file_name)

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    """Load semua model dengan error handling."""
    face_cascade = None
    age_net = None
    gender_net = None
    emotion_net = None
    
    # 1. Face Cascade
    try:
        path = str(MODELS_TO_DOWNLOAD["haarcascade_frontalface_default.xml"][1])
        face_cascade = cv2.CascadeClassifier(path)
    except Exception as e:
        st.error(f"Gagal load Face Cascade: {e}")
    
    # 2. Age Net
    try:
        model_path = str(MODELS_TO_DOWNLOAD["age_net.caffemodel"][1])
        proto_path = str(MODELS_TO_DOWNLOAD["age_deploy.prototxt"][1])
        age_net = cv2.dnn.readNet(model_path, proto_path)
    except Exception as e:
        st.warning(f"Model Usia tidak tersedia: {e}")
    
    # 3. Gender Net
    try:
        model_path = str(MODELS_TO_DOWNLOAD["gender_net.caffemodel"][1])
        proto_path = str(MODELS_TO_DOWNLOAD["gender_deploy.prototxt"][1])
        gender_net = cv2.dnn.readNet(model_path, proto_path)
    except Exception as e:
        st.warning(f"Model Gender tidak tersedia: {e}")
    
    # 4. Emotion Net
    try:
        path = str(MODELS_TO_DOWNLOAD["emotion-ferplus-8.onnx"][1])
        emotion_net = cv2.dnn.readNetFromONNX(path)
    except Exception as e:
        st.warning(f"Model Emosi tidak tersedia: {e}")
    
    return face_cascade, age_net, gender_net, emotion_net

face_cascade, age_net, gender_net, emotion_net = load_models()

# Konstanta
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Pria', 'Wanita']
EMOTION_LIST = ['Netral', 'Senang', 'Kaget', 'Sedih', 'Marah', 'Jijik', 'Takut', 'Penghinaan']

# --- FUNGSI PEMROSESAN GAMBAR ---
def process_uploaded_image(uploaded_file):
    """Memproses gambar yang diupload user."""
    # Baca gambar
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Convert RGB ke BGR (OpenCV format)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_array
    
    # Deteksi wajah
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    
    results = []
    for (x, y, w, h) in faces:
        face_roi = img_bgr[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (96, 96))
        blob = cv2.dnn.blobFromImage(face_resized, 1.0, (96, 96), (104, 117, 123), swapRB=False)
        
        gender = "Unknown"
        age = "Unknown"
        emotion = "Unknown"
        age_conf = 0.0
        
        # Prediksi Gender
        if gender_net is not None:
            try:
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender_idx = np.argmax(gender_preds[0])
                gender = GENDER_LIST[gender_idx] if gender_idx < len(GENDER_LIST) else "Unknown"
            except Exception:
                pass
        
        # Prediksi Usia
        if age_net is not None:
            try:
                blob_color = cv2.dnn.blobFromImage(face_resized, 1.0, (96, 96), (78, 87, 114), swapRB=False)
                age_net.setInput(blob_color)
                age_preds = age_net.forward()
                
                if len(age_preds.shape) == 4:
                    age_scores = np.mean(age_preds, axis=(2, 3))[0]
                else:
                    age_scores = age_preds[0]
                
                age_idx = np.argmax(age_scores)
                if 0 <= age_idx < len(AGE_BUCKETS):
                    age = AGE_BUCKETS[age_idx]
                    age_conf = age_scores[age_idx]
            except Exception:
                pass
        
        # Prediksi Emosi
        if emotion_net is not None:
            try:
                face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                face_resized_emo = cv2.resize(face_gray, (64, 64))
                face_normalized = face_resized_emo.astype(np.float32) / 255.0
                face_input = np.expand_dims(np.expand_dims(face_normalized, axis=0), axis=0)
                
                emotion_net.setInput(face_input)
                emotion_preds = emotion_net.forward()
                emotion_idx = np.argmax(emotion_preds[0])
                emotion = EMOTION_LIST[emotion_idx] if emotion_idx < len(EMOTION_LIST) else "Unknown"
            except Exception:
                pass
        
        # Gambar kotak & label
        cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 255), 3)
        conf_str = f"({age_conf*100:.0f}%)" if age != "Unknown" else ""
        label_top = f"{gender}, {age} {conf_str}"
        label_bottom = f"{emotion}"
        cv2.putText(img_bgr, label_top, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(img_bgr, label_bottom, (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        results.append({
            "gender": gender,
            "age": age,
            "age_conf": float(age_conf),
            "emotion": emotion
        })
    
    # Convert kembali ke RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb, results

# --- UI STREAMLIT ---
st.title("🤖 Aplikasi Deteksi Wajah, Usia, Gender & Emosi")

st.info("""
☁️ **Versi Cloud - Image Upload Mode**

Upload foto wajah untuk mendapatkan analisis AI tentang usia, gender, dan emosi!

💻 **Ingin versi real-time webcam?** Clone repo ini dan jalankan di lokal:
```bash
git clone https://github.com/ketsar28/real-time-facial-recognition-app.git
cd real-time-facial-recognition-app
pip install -r requirements.txt
streamlit run webapp_face_detection.py
```
""")

st.header("📤 Upload Gambar untuk Deteksi")

uploaded_file = st.file_uploader(
    "Pilih gambar wajah (JPG, PNG, JPEG)", 
    type=['jpg', 'png', 'jpeg'],
    help="Upload foto dengan wajah yang terlihat jelas untuk hasil terbaik"
)

if uploaded_file is not None:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("🖼️ Hasil Deteksi")
        processed_img, results = process_uploaded_image(uploaded_file)
        st.image(processed_img, use_column_width=True, caption="Gambar dengan Deteksi AI")
    
    with col2:
        st.subheader("📊 Analisis")
        st.metric("Jumlah Wajah", len(results))
        
        if not results:
            st.warning("⚠️ Tidak ada wajah terdeteksi. Coba foto lain dengan wajah yang lebih jelas!")
        else:
            for i, result in enumerate(results):
                st.markdown("---")
                st.write(f"**Wajah #{i+1}**")
                st.write(f"👤 **Gender:** {result['gender']}")
                conf_val = result.get('age_conf', 0.0) * 100
                st.write(f"🎂 **Usia:** {result['age']} ({conf_val:.1f}%)")
                st.write(f"😊 **Emosi:** {result['emotion']}")
else:
    st.info("👆 Upload gambar untuk memulai deteksi!")

# Sidebar
st.sidebar.title("Tentang Aplikasi")
st.sidebar.info("""
Aplikasi ini menggunakan Computer Vision dan Deep Learning untuk mendeteksi:
- 👤 Gender (Pria/Wanita)
- 🎂 Usia (8 rentang usia)
- 😊 Emosi (8 jenis emosi)

**Model yang digunakan:**
- Haar Cascade (Face Detection)
- Caffe Model (Age & Gender)
- ONNX FER+ (Emotion Recognition)
""")

st.sidebar.markdown("---")
st.sidebar.caption("Dibuat oleh KETSAR | 2026")
