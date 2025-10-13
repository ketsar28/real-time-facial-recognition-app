# 🤖 Real-Time Face Analysis Web App

Aplikasi web canggih yang dibangun menggunakan Streamlit dan OpenCV untuk melakukan analisis wajah secara *real-time* langsung dari browser kamu. Aplikasi ini mampu mendeteksi wajah, kemudian memperkirakan usia, gender, dan ekspresi emosi secara bersamaan.

---

## 🌟 Fitur Utama

-   **Deteksi Wajah Real-Time**: Mendeteksi satu atau beberapa wajah secara bersamaan dari streaming webcam.
-   **Estimasi Usia**: Memperkirakan rentang usia dari setiap wajah yang terdeteksi.
-   **Prediksi Gender**: Mengenali gender (Pria/Wanita).
-   **Analisis Emosi**: Mengenali 8 ekspresi emosi dasar (Netral, Senang, Kaget, Sedih, Marah, Jijik, Takut, Penghinaan).
-   **Antarmuka Interaktif**: Dibangun dengan Streamlit untuk pengalaman pengguna yang mulus dan responsif.
-   **Desain Modern**: Tampilan antarmuka bertema gelap yang elegan dan nyaman di mata.

---

## 🛠️ Teknologi yang Digunakan

-   **Bahasa**: Python
-   **Framework Web**: Streamlit
-   **Computer Vision**: OpenCV
-   **Streaming Video**: Streamlit-WebRTC
-   **Numerik**: NumPy
-   **Model AI**:
    -   Haar Cascades (Deteksi Wajah)
    -   Caffe Models (Usia & Gender)
    -   ONNX Model (Emosi)

---

## 🚀 Cara Menjalankan Proyek Secara Lokal

Ikuti langkah-langkah di bawah ini untuk menjalankan aplikasi di komputer kamu.

### 1. Prasyarat

-   Pastikan sudah menginstal [Python](https://www.python.org/downloads/) versi 3.8 atau yang lebih baru.
-   Disarankan menggunakan manajer paket seperti [Anaconda](https://www.anaconda.com/products/distribution) untuk mengelola *environment*.

### 2. Kloning Repositori

Buka terminal atau Git Bash dan kloning repositori ini:
```bash
git clone https://github.com/ketsar28/real-time-facial-recognition-app.git
cd real-time-facial-recognition-app
