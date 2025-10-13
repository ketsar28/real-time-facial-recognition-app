# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 14:53:39 2025

@author: KETSAR
"""

import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import threading
import os
import requests
from pathlib import Path

# --- KONFIGURASI PAGE & STYLE ---
st.set_page_config(layout="wide", page_title="Aplikasi Deteksi Wajah Canggih")

st.markdown("""
<style>
    .main { background-color: #0E1117; }
    h1 { color: #00A8E8; text-align: center; }
    h2, h3 { color: #FFFFFF; text-align: center; }
    .sidebar .sidebar-content { background-color: #161A25; }
    .stMetric { background-color: #262730; border-radius: 10px; padding: 10px; text-align: center; }
    .stMetricLabel { font-weight: bold; color: #00A8E8; }
    .stMetricValue { font-size: 2em; color: #FFFFFF; }
</style>
""", unsafe_allow_html=True)

# --- FUNGSI HELPER UNTUK MENGUNDUH MODEL ---
def download_file(url, file_path, file_name):
    """Mengunduh file dari URL dengan progress bar Streamlit."""
    local_path = Path(file_path)
    if not local_path.exists():
        with st.spinner(f"Mengunduh model {file_name}... (Ini hanya terjadi sekali)"):
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status() # Cek jika ada error HTTP
                
                total_size = int(response.headers.get('content-length', 0))
                progress_bar = st.progress(0)
                
                with open(local_path, "wb") as f:
                    downloaded_size = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        progress = min(int(100 * downloaded_size / total_size), 100) if total_size > 0 else 0
                        progress_bar.progress(progress)
                
                progress_bar.empty() # Hapus progress bar setelah selesai
            except requests.exceptions.RequestException as e:
                st.error(f"Gagal mengunduh {file_name}: {e}")
                st.stop() # Hentikan aplikasi jika model gagal diunduh

# --- MODEL PREPROCESSING DAN PATH ---
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True) # Buat folder 'model' jika belum ada

# Definisikan URL dan path lokal untuk setiap model
MODELS_TO_DOWNLOAD = {
    "haarcascade_frontalface_default.xml": ("https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml", Path("haarcascade_frontalface_default.xml")),
    "age_deploy.prototxt": ("https://drive.google.com/uc?export=download&id=1rrG6YaTl5qG9fPetzWATx0x2r35CZuMR", MODEL_DIR / "age_deploy.prototxt"),
    "age_net.caffemodel": ("https://drive.google.com/uc?export=download&id=1OghpY-948_suElSubg8gLmexyRRVIoKV", MODEL_DIR / "age_net.caffemodel"),
    "gender_deploy.prototxt": ("https://drive.google.com/uc?export=download&id=18Bi0V4qy0vUScu74S_o6_WJKIDb0KHXw", MODEL_DIR / "gender_deploy.prototxt"),
    "gender_net.caffemodel": ("https://drive.google.com/uc?export=download&id=1lfCdxRtrmO38C9KhnCsvQ91bJS51QVHo", MODEL_DIR / "gender_net.caffemodel"),
    "emotion-ferplus-8.onnx": ("https://drive.google.com/uc?export=download&id=1ECzQ0SJ7DD46y1sI_IoSKcluUmbn9LW7", MODEL_DIR / "emotion-ferplus-8.onnx")
}

# download untuk setiap model
for file_name, (url, local_path) in MODELS_TO_DOWNLOAD.items():
    download_file(url, local_path, file_name)

# --- FUNGSI UNTUK NGELOAD MODEL (DENGAN CACHING) ---
@st.cache_resource
def load_models():
    """Memuat semua model AI dari path lokal."""
    face_cascade = cv2.CascadeClassifier(str(MODELS_TO_DOWNLOAD["haarcascade_frontalface_default.xml"][1]))
    age_net = cv2.dnn.readNet(str(MODELS_TO_DOWNLOAD["age_net.caffemodel"][1]), str(MODELS_TO_DOWNLOAD["age_deploy.prototxt"][1]))
    gender_net = cv2.dnn.readNet(str(MODELS_TO_DOWNLOAD["gender_net.caffemodel"][1]), str(MODELS_TO_DOWNLOAD["gender_deploy.prototxt"][1]))
    emotion_net = cv2.dnn.readNetFromONNX(str(MODELS_TO_DOWNLOAD["emotion-ferplus-8.onnx"][1]))
    print("Semua model berhasil dimuat dari file lokal.")
    return face_cascade, age_net, gender_net, emotion_net

face_cascade, age_net, gender_net, emotion_net = load_models()

# Daftar konstantax
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Pria', 'Wanita']
EMOTION_LIST = ['Netral', 'Senang', 'Kaget', 'Sedih', 'Marah', 'Jijik', 'Takut', 'Penghinaan']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# --- KELAS UTAMA UNTUK VIDEO PROCESSING ---
class FaceDetector(VideoTransformerBase):
    def __init__(self):
        self.frame_lock = threading.Lock()
        self.latest_results = []

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        current_results = []
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_img_color = img[y:y + h, x:x + w]
            blob_color = cv2.dnn.blobFromImage(face_img_color, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            gender_net.setInput(blob_color)
            gender = GENDER_LIST[gender_net.forward()[0].argmax()]
            age_net.setInput(blob_color)
            age = AGE_BUCKETS[age_net.forward()[0].argmax()]
            face_roi_gray = gray[y:y + h, x:x + w]
            resized_face = cv2.resize(face_roi_gray, (64, 64))
            reshaped_face = resized_face.reshape(1, 1, 64, 64)
            emotion_net.setInput(reshaped_face)
            emotion = EMOTION_LIST[emotion_net.forward()[0].argmax()]
            current_results.append({"gender": gender, "age": age, "emotion": emotion})
            label_age_gender = f"{gender}, {age}"
            cv2.putText(img, label_age_gender, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(img, emotion, (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        with self.frame_lock:
            self.latest_results = current_results
        return img

# --- MEMBANGUN UI STREAMLIT ---
st.title("🤖 Aplikasi Deteksi Wajah, Usia, Gender & Emosi")
st.markdown("Arahkan wajah Anda ke webcam untuk melihat keajaiban AI secara *real-time*!")
col1, col2 = st.columns(spec=[3, 1])
with col1:
    st.header("Streaming Webcam")
    webrtc_ctx = webrtc_streamer(key="face-detection", video_processor_factory=FaceDetector, media_stream_constraints={"video": True, "audio": False}, rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
with col2:
    st.header("Hasil Analisis")
    face_count_placeholder = st.empty()
    results_placeholder = st.empty()
    if not webrtc_ctx.state.playing:
        st.info("Klik 'START' untuk menyalakan kamera.")
    else:
        st.success("Kamera aktif! Menunggu deteksi...")
    while webrtc_ctx.state.playing:
        if webrtc_ctx.video_processor:
            with webrtc_ctx.video_processor.frame_lock:
                results = webrtc_ctx.video_processor.latest_results
            face_count_placeholder.metric("Jumlah Wajah Terdeteksi", len(results))
            with results_placeholder.container():
                if not results:
                    st.info("Tidak ada wajah yang terdeteksi.")
                else:
                    for i, result in enumerate(results):
                        st.markdown(f"---")
                        st.subheader(f"Wajah #{i+1}")
                        st.write(f"**Gender:** {result['gender']}")
                        st.write(f"**Estimasi Usia:** {result['age']}")
                        st.write(f"**Emosi:** {result['emotion']}")
        else:
            break
st.sidebar.title("Tentang Aplikasi")
st.sidebar.info("Ini adalah aplikasi web yang dibangun dengan Streamlit untuk mendemonstrasikan kemampuan Computer Vision dalam mendeteksi wajah, usia, gender, dan emosi secara real-time menggunakan model-model AI dari OpenCV.")