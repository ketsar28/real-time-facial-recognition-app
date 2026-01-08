# Base image: Python 3.10 (kompatibel dengan aiortc/WebRTC)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies untuk OpenCV & PyAV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements & install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file aplikasi
COPY . .

# Railway menyediakan $PORT variable, jadi kita HARUS pakai port dinamis
# Default ke 8501 kalau $PORT ga ada (untuk local testing)
ENV PORT=8501

# Jalankan Streamlit dengan konfigurasi Railway
CMD streamlit run webapp_face_detection.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.serverAddress="0.0.0.0"