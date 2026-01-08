# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 14:53:39 2025
Optimized on Jan 08 2026

@author: KETSAR
"""

import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import threading
import os
import requests
from pathlib import Path
import time

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
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                progress_bar = st.progress(0)
                
                with open(local_path, "wb") as f:
                    downloaded_size = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        progress = min(int(100 * downloaded_size / total_size), 100) if total_size > 0 else 0
                        progress_bar.progress(progress)
                
                progress_bar.empty()
            except requests.exceptions.RequestException as e:
                st.error(f"Gagal mengunduh {file_name}: {e}")
                st.stop()

# --- MODEL PREPROCESSING DAN PATH ---
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

# Definisikan URL dan path lokal untuk setiap model
MODELS_TO_DOWNLOAD = {
    "haarcascade_frontalface_default.xml": ("https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml", Path("haarcascade_frontalface_default.xml")),
    # Gunakan URL Mirror yang valid untuk Age & Gender (smahesh29 repo)
    "age_deploy.prototxt": ("https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/age_deploy.prototxt", MODEL_DIR / "age_deploy.prototxt"),
    "age_net.caffemodel": ("https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/age_net.caffemodel", MODEL_DIR / "age_net.caffemodel"),
    "gender_deploy.prototxt": ("https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/gender_deploy.prototxt", MODEL_DIR / "gender_deploy.prototxt"),
    "gender_net.caffemodel": ("https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/gender_net.caffemodel", MODEL_DIR / "gender_net.caffemodel"),
    # Gunakan URL Official ONNX (RAW) - Pastikan 'raw', bukan 'blob'
    "emotion-ferplus-8.onnx": ("https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx", MODEL_DIR / "emotion-ferplus-8.onnx")
}

# Download untuk setiap model
for file_name, (url, local_path) in MODELS_TO_DOWNLOAD.items():
    download_file(url, local_path, file_name)

# --- FUNGSI UNTUK NGELOAD MODEL (DENGAN CACHING) ---
@st.cache_resource
def load_models():
    """Memuat semua model AI dari path lokal dengan Error Handling Robust."""
    face_cascade = None
    age_net = None
    gender_net = None
    emotion_net = None

    # 1. Face Cascade
    try:
        path = str(MODELS_TO_DOWNLOAD["haarcascade_frontalface_default.xml"][1])
        face_cascade = cv2.CascadeClassifier(path)
        if face_cascade.empty():
             st.error("Gagal memuat Face Cascade XML!")
    except Exception as e:
        st.error(f"Error loading Face Cascade: {e}")

    # 2. Age Net
    try:
        path_model = str(MODELS_TO_DOWNLOAD["age_net.caffemodel"][1])
        path_proto = str(MODELS_TO_DOWNLOAD["age_deploy.prototxt"][1])
        age_net = cv2.dnn.readNet(path_model, path_proto)
    except Exception as e:
        print(f"⚠️ Gagal memuat Age Net: {e}")
        st.warning("Model Usia gagal dimuat. Fitur prediksi usia dimatikan.")

    # 3. Gender Net
    try:
        path_model = str(MODELS_TO_DOWNLOAD["gender_net.caffemodel"][1])
        path_proto = str(MODELS_TO_DOWNLOAD["gender_deploy.prototxt"][1])
        gender_net = cv2.dnn.readNet(path_model, path_proto)
    except Exception as e:
        print(f"⚠️ Gagal memuat Gender Net: {e}")
        st.warning("Model Gender gagal dimuat. Fitur prediksi gender dimatikan.")

    # 4. Emotion Net
    try:
        path = str(MODELS_TO_DOWNLOAD["emotion-ferplus-8.onnx"][1])
        emotion_net = cv2.dnn.readNetFromONNX(path)
    except Exception as e:
        print(f"⚠️ Gagal memuat Emotion Net: {e}")
        st.warning("Model Emosi gagal dimuat (cek koneksi/file). Fitur emosi dimatikan.")

    print("✅ Proses pemuatan model selesai (Partial/Full).")
    return face_cascade, age_net, gender_net, emotion_net

face_cascade, age_net, gender_net, emotion_net = load_models()

# Daftar konstanta
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Pria', 'Wanita']
EMOTION_LIST = ['Netral', 'Senang', 'Kaget', 'Sedih', 'Marah', 'Jijik', 'Takut', 'Penghinaan']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# --- KONFIGURASI PERFORMA ---
# Frame skip: proses 1 dari setiap N frame untuk performa lebih baik
FRAME_SKIP = 3  # Proses setiap 3 frame (hemat resource)
DETECTION_SCALE = 0.5  # Resize frame untuk deteksi lebih cepat


# --- KELAS UTAMA UNTUK VIDEO PROCESSING (UPDATED TO recv()) ---
# --- KELAS UTAMA UNTUK VIDEO PROCESSING (UPDATED TO recv()) ---
class FaceDetector(VideoProcessorBase):
    """
    Video processor untuk deteksi wajah real-time.
    Menggunakan recv() method (pengganti transform() yang deprecated).
    """
    
    def __init__(self):
        self.frame_lock = threading.Lock()
        self.latest_results = []
        self.frame_count = 0  # Counter untuk frame skipping
    
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        Method utama untuk memproses setiap frame video.
        """
        # Konversi frame ke numpy array (BGR format)
        img = frame.to_ndarray(format="bgr24")
        
        # Frame skipping untuk performa lebih baik
        self.frame_count += 1
        
        # Hanya proses deteksi baru setiap N frame
        if self.frame_count % FRAME_SKIP == 0:
            self._process_frame(img)
        
        # SELALU gambar hasil terakhir di frame (agar tidak flickering)
        img = self._draw_cached_results(img)
        
        # Konversi kembali ke av.VideoFrame
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def _process_frame(self, img: np.ndarray):
        """Proses frame untuk deteksi wajah, gender, usia, dan emosi."""
        # 0. Safety Check: Pastikan Face Cascade dimuat
        if face_cascade is None:
            return

        # Resize untuk deteksi lebih cepat
        small_frame = cv2.resize(img, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Deteksi wajah pada frame yang sudah di-resize
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(20, 20)
        )
        
        current_results = []
        
        for (x, y, w, h) in faces:
            try:
                # Scale koordinat kembali ke ukuran asli
                x_orig = int(x / DETECTION_SCALE)
                y_orig = int(y / DETECTION_SCALE)
                w_orig = int(w / DETECTION_SCALE)
                h_orig = int(h / DETECTION_SCALE)
                
                # Ambil ROI wajah dari frame asli untuk analisis
                face_img_color = img[y_orig:y_orig + h_orig, x_orig:x_orig + w_orig]
                
                if face_img_color.size == 0:
                    continue
                
                # Default values jika prediksi gagal
                gender = "Unknown"
                age = "Unknown" 
                emotion = "Unknown"
                age_conf = 0.0

                # Standardize Input Blob (Common for Age/Gender)
                blob_color = cv2.dnn.blobFromImage(
                    face_img_color, 1.0, (227, 227), 
                    MODEL_MEAN_VALUES, swapRB=False
                )

                # --- 1. PREDIKSI GENDER ---
                if gender_net is not None:
                    try:
                        gender_net.setInput(blob_color)
                        gender_preds = gender_net.forward()
                        gender_idx = gender_preds[0].argmax()
                        
                        if 0 <= gender_idx < len(GENDER_LIST):
                            gender = GENDER_LIST[gender_idx]
                    except Exception as e:
                        print(f"Propagated Error (Gender): {e}")

                # --- 2. PREDIKSI USIA ---
                if age_net is not None:
                    try:
                        age_net.setInput(blob_color)
                        age_preds = age_net.forward()
                        
                        # Logika Penanganan Output Multi-Dimensi (Robust)
                        if age_preds.ndim == 4:
                             age_scores = np.mean(age_preds, axis=(2, 3))
                             age_idx = age_scores[0].argmax()
                             age_conf = age_scores[0][age_idx] # Ambil confidence
                        elif age_preds.ndim == 2:
                            age_idx = age_preds[0].argmax()
                            age_conf = age_preds[0][age_idx]
                        else:
                            age_flat = age_preds.flatten()
                            age_idx = age_flat.argmax()
                            age_conf = age_flat[age_idx]

                        if 0 <= age_idx < len(AGE_BUCKETS):
                            age = AGE_BUCKETS[age_idx]
                    except Exception as e:
                        print(f"Propagated Error (Age): {e}")

                # --- 3. PREDIKSI EMOSI ---
                if emotion_net is not None:
                    try:
                        face_roi_gray = cv2.cvtColor(face_img_color, cv2.COLOR_BGR2GRAY)
                        resized_face = cv2.resize(face_roi_gray, (64, 64))
                        reshaped_face = resized_face.reshape(1, 1, 64, 64).astype(np.float32)
                        
                        emotion_net.setInput(reshaped_face)
                        emotion_preds = emotion_net.forward()
                        emotion_idx = emotion_preds[0].argmax()
                        
                        if 0 <= emotion_idx < len(EMOTION_LIST):
                             emotion = EMOTION_LIST[emotion_idx]
                    except Exception as e:
                        print(f"Propagated Error (Emotion): {e}")
                
                # Simpan hasil (Face tetap disimpan meski prediksi attribute gagal)
                current_results.append({
                    "gender": gender, 
                    "age": age, 
                    "age_conf": float(age_conf), # Simpan confidence
                    "emotion": emotion,
                    "bbox": (x_orig, y_orig, w_orig, h_orig)
                })
                           
            except Exception as e:
                print(f"⚠️ Critical Error processing face loop: {e}")
                continue
        
        # Update hasil dengan thread-safe
        with self.frame_lock:
            self.latest_results = current_results
    
    def _draw_cached_results(self, img: np.ndarray) -> np.ndarray:
        """Gambar hasil deteksi terakhir pada frame."""
        with self.frame_lock:
            results = self.latest_results.copy()
        
        for result in results:
            x, y, w, h = result["bbox"]
            gender = result["gender"]
            age = result["age"]
            emotion = result["emotion"]
            age_conf = result.get("age_conf", 0.0)

            # Warna kotak
            color = (0, 255, 0) # Hijau
            
            # Gambar kotak
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            # Label background logic (opsional, bisa ditambah cv2.rectangle fill)
            
            # Format Label: "Pria, 25-30 (85%)"
            conf_str = f"({age_conf*100:.0f}%)" if age != "Unknown" else ""
            label_top = f"{gender}, {age} {conf_str}"
            label_bottom = f"{emotion}"
            
            # Label Atas (Gender, Age)
            cv2.putText(img, label_top, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Label Bawah (Emotion)
            cv2.putText(img, label_bottom, (x, y + h + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return img


# --- KONFIGURASI RTC (STUN/TURN SERVERS) ---
# Menggunakan server STUN yang GRATIS dan AKTIF
# TURN server: numb.viagenie.ca sudah DISCONTINUED, jadi kita pakai STUN only
RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        # Google STUN servers (gratis, reliable)
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        
        # Free TURN Server (OpenRelay) - "Penyelamat" jika STUN gagal
        # Ini penting untuk deployment di Streamlit Cloud!
        {"urls": ["turn:openrelay.metered.ca:80"], "username": "openrelayproject", "credential": "openrelayproject"},
        {"urls": ["turn:openrelay.metered.ca:443"], "username": "openrelayproject", "credential": "openrelayproject"},
        {"urls": ["turn:openrelay.metered.ca:443?transport=tcp"], "username": "openrelayproject", "credential": "openrelayproject"},
    ]
})


# --- MEMBANGUN UI STREAMLIT ---
st.title("🤖 Aplikasi Deteksi Wajah, Usia, Gender & Emosi")
st.markdown("Arahkan wajah Anda ke webcam untuk melihat keajaiban AI secara *real-time*!")

# Info box tentang performa
with st.expander("ℹ️ Tips Performa", expanded=False):
    st.markdown("""
    - **Frame Skip**: Aplikasi memproses 1 dari setiap 3 frame untuk performa optimal
    - **Jika lambat**: Pastikan tidak ada aplikasi berat lain yang berjalan
    - **Jika kamera tidak muncul**: Coba refresh halaman atau gunakan browser Chrome
    """)

col1, col2 = st.columns(spec=[3, 1])

with col1:
    st.header("Streaming Webcam")
    webrtc_ctx = webrtc_streamer(
        key="face-detection",
        video_processor_factory=FaceDetector,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},  # Batasi resolusi untuk performa
                "height": {"ideal": 480},
                "frameRate": {"ideal": 15, "max": 20}  # Batasi FPS
            }, 
            "audio": False
        },
        rtc_configuration=RTC_CONFIG,
        async_processing=True,  # Proses async untuk performa lebih baik
    )

with col2:
    st.header("Hasil Analisis")
    face_count_placeholder = st.empty()
    results_placeholder = st.empty()
    
    if webrtc_ctx.state.playing:
        st.success("🟢 Kamera aktif!")
        
        # Loop untuk update hasil deteksi
        while webrtc_ctx.state.playing:
            if webrtc_ctx.video_processor:
                with webrtc_ctx.video_processor.frame_lock:
                    results = webrtc_ctx.video_processor.latest_results.copy()
                
                face_count_placeholder.metric("Jumlah Wajah Terdeteksi", len(results))
                
                with results_placeholder.container():
                    if not results:
                        st.info("👀 Menunggu deteksi wajah...")
                    else:
                        for i, result in enumerate(results):
                            st.markdown("---")
                            st.subheader(f"Wajah #{i+1}")
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.write(f"**Gender:** {result['gender']}")
                                # Tambahkan probabilitas di sini
                                conf_val = result.get('age_conf', 0.0) * 100
                                st.write(f"**Usia:** {result['age']} ({conf_val:.1f}%)")
                            with col_b:
                                st.write(f"**Emosi:** {result['emotion']}")
            else:
                break
            
            # Small delay untuk mengurangi CPU usage
            time.sleep(0.1)
    else:
        st.info("👆 Klik 'START' untuk menyalakan kamera.")

# --- SIDEBAR ---
st.sidebar.title("Tentang Aplikasi")
st.sidebar.info("""
Ini adalah aplikasi web yang dibangun dengan Streamlit untuk mendemonstrasikan 
kemampuan Computer Vision dalam mendeteksi wajah, usia, gender, dan emosi 
secara real-time menggunakan model-model AI dari OpenCV.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Pengaturan Performa")
st.sidebar.caption(f"""
- **Frame Skip**: {FRAME_SKIP} (proses 1 dari {FRAME_SKIP} frame)
- **Detection Scale**: {DETECTION_SCALE} (resize untuk deteksi)
- **Target FPS**: 15-20
""")
