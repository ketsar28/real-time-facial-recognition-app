# 🎭 Real-Time Face Analysis Web Application

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-FF4B4B.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Aplikasi web canggih berbasis AI untuk analisis wajah secara real-time**

[Demo](#-demo) • [Fitur](#-fitur-utama) • [Instalasi](#-instalasi) • [Penggunaan](#-cara-penggunaan) • [Teknologi](#-teknologi)

</div>

---

## 📖 Tentang Project

Real-Time Face Analysis adalah aplikasi web interaktif yang memanfaatkan kekuatan Computer Vision dan Deep Learning untuk melakukan analisis wajah secara real-time langsung dari webcam browser Anda. Aplikasi ini dapat mendeteksi wajah, memperkirakan usia, memprediksi gender, dan menganalisis ekspresi emosi secara bersamaan dengan performa tinggi dan akurasi yang optimal.

### 🎯 Keunggulan

- ⚡ **Real-Time Processing** - Analisis instan tanpa delay
- 🎨 **Modern UI/UX** - Interface yang elegan dan user-friendly
- 🔒 **Privacy First** - Semua processing dilakukan di sisi client
- 📊 **Multi-Task Analysis** - Deteksi wajah, usia, gender, dan emosi sekaligus
- 🌐 **Web-Based** - Tidak perlu instalasi software tambahan
- 🚀 **High Performance** - Optimized untuk performa maksimal

---

## ✨ Fitur Utama

### 🔍 Deteksi Wajah Real-Time
Mendeteksi satu atau beberapa wajah secara bersamaan dari streaming webcam menggunakan Haar Cascades dengan akurasi tinggi.

### 👤 Estimasi Usia
Memperkirakan rentang usia dari setiap wajah yang terdeteksi menggunakan pre-trained Caffe model dengan klasifikasi multi-kategori:
- 0-2 tahun
- 4-6 tahun
- 8-12 tahun
- 15-20 tahun
- 25-32 tahun
- 38-43 tahun
- 48-53 tahun
- 60-100 tahun

### 🚻 Prediksi Gender
Mengenali gender (Pria/Wanita) dengan tingkat akurasi tinggi menggunakan deep learning model berbasis Caffe.

### 😊 Analisis Emosi
Mengenali 8 ekspresi emosi dasar dengan menggunakan model ONNX:
- 😐 Netral (Neutral)
- 😊 Senang (Happy)
- 😲 Kaget (Surprise)
- 😢 Sedih (Sad)
- 😠 Marah (Angry)
- 🤢 Jijik (Disgust)
- 😨 Takut (Fear)
- 😒 Penghinaan (Contempt)

### 🎨 Antarmuka Interaktif
- Interface modern dengan tema gelap yang nyaman di mata
- Real-time metrics dan statistics
- Responsive design untuk berbagai ukuran layar
- Visual feedback yang informatif

---

## 🛠️ Teknologi

### Core Technologies
| Teknologi | Versi | Kegunaan |
|-----------|-------|----------|
| **Python** | 3.8+ | Bahasa pemrograman utama |
| **Streamlit** | 1.50.0 | Framework web application |
| **OpenCV** | 4.x | Computer vision dan image processing |
| **TensorFlow** | 2.20.0 | Deep learning framework |
| **NumPy** | 2.1.3 | Operasi numerik dan array |
| **Streamlit-WebRTC** | 0.63.11 | Real-time video streaming |

### AI Models
- **Haar Cascades** - Face detection
- **Caffe Models** - Age & gender prediction
- **ONNX Model** - Emotion recognition

---

## 📦 Instalasi

### Prasyarat

Pastikan sistem Anda memiliki:
- Python 3.8 atau lebih tinggi
- pip (Python package manager)
- Webcam yang berfungsi
- Koneksi internet (untuk download model pertama kali)

### Langkah Instalasi

1. **Clone repository**
   ```bash
   git clone https://github.com/ketsar28/real-time-facial-recognition-app.git
   cd real-time-facial-recognition-app
   ```

2. **Buat virtual environment** (opsional tapi direkomendasikan)
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download model files** (otomatis saat pertama kali running)

   Model akan otomatis diunduh saat pertama kali menjalankan aplikasi:
   - `haarcascade_frontalface_default.xml`
   - `age_net.caffemodel` & `age_deploy.prototxt`
   - `gender_net.caffemodel` & `gender_deploy.prototxt`
   - `emotion-ferplus-8.onnx`

---

## 🚀 Cara Penggunaan

### Menjalankan Aplikasi

1. **Jalankan aplikasi Streamlit**
   ```bash
   streamlit run webapp_face_detection.py
   ```

2. **Akses aplikasi**

   Browser akan otomatis terbuka di `http://localhost:8501`

   Atau buka secara manual di browser favorit Anda.

3. **Izinkan akses webcam**

   Browser akan meminta izin untuk mengakses webcam. Klik **Allow/Izinkan**.

4. **Mulai analisis**

   Aplikasi akan otomatis mulai mendeteksi dan menganalisis wajah Anda secara real-time!

### Tips Penggunaan

- 🔆 **Pencahayaan** - Gunakan pencahayaan yang cukup untuk hasil optimal
- 📏 **Jarak** - Posisikan wajah pada jarak 50-100cm dari kamera
- 👥 **Multi-Face** - Aplikasi dapat mendeteksi beberapa wajah sekaligus
- 🎭 **Ekspresi** - Coba berbagai ekspresi untuk melihat deteksi emosi

---

## 📁 Struktur Project

```
real-time-facial-recognition-app/
│
├── webapp_face_detection.py    # Main application file
├── requirements.txt             # Python dependencies
├── README.md                    # Documentation
│
└── models/                      # AI models (auto-downloaded)
    ├── haarcascade_frontalface_default.xml
    ├── age_net.caffemodel
    ├── age_deploy.prototxt
    ├── gender_net.caffemodel
    ├── gender_deploy.prototxt
    └── emotion-ferplus-8.onnx
```

---

## 🎬 Demo

### Screenshots

*Aplikasi sedang menganalisis wajah secara real-time dengan informasi usia, gender, dan emosi*

### Live Demo

Coba aplikasi secara langsung: [Streamlit Cloud Demo](https://share.streamlit.io/user/ketsar28)

---

## 🔧 Troubleshooting

### Webcam tidak terdeteksi
```python
# Pastikan tidak ada aplikasi lain yang menggunakan webcam
# Cek permission browser untuk akses webcam
# Restart browser dan coba lagi
```

### Error saat instalasi dependencies
```bash
# Upgrade pip terlebih dahulu
pip install --upgrade pip

# Install ulang requirements
pip install -r requirements.txt --no-cache-dir
```

### Model gagal download
```bash
# Pastikan koneksi internet stabil
# Hapus file model yang corrupt di folder models/
# Restart aplikasi untuk re-download
```

---

## 🚀 Deployment

### Deploy ke Streamlit Cloud

1. Fork repository ini
2. Login ke [Streamlit Cloud](https://streamlit.io/cloud)
3. Klik "New app"
4. Pilih repository Anda
5. Set main file: `webapp_face_detection.py`
6. Deploy!

### Deploy ke Heroku/Railway/Render

Ikuti dokumentasi deployment masing-masing platform dan pastikan:
- Tambahkan `Procfile` jika diperlukan
- Set buildpack Python
- Konfigurasi environment variables jika ada

---

## 🤝 Kontribusi

Kontribusi sangat diterima! Jika Anda ingin berkontribusi:

1. Fork repository ini
2. Buat branch baru (`git checkout -b feature/AmazingFeature`)
3. Commit perubahan Anda (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buat Pull Request

---

## 🐛 Bug Reports & Feature Requests

Menemukan bug atau punya ide fitur baru? Silakan buat [Issue](https://github.com/ketsar28/real-time-facial-recognition-app/issues) di repository ini.

---

## 📚 Resources & References

- [Streamlit Documentation](https://docs.streamlit.io/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [FER+ Dataset](https://github.com/microsoft/FERPlus)

---

## 📜 License

Project ini dilisensikan di bawah [MIT License](LICENSE) - lihat file LICENSE untuk detail lengkap.

---

## 👤 Author

**Ketsar Ali**

Dibuat dengan ❤️ oleh Ketsar Ali - Passionate AI/ML Engineer & Computer Vision Enthusiast

### 🌐 Connect With Me

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-ketsar28-181717?style=for-the-badge&logo=github)](https://github.com/ketsar28/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-ketsarali-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/ketsarali/)
[![Instagram](https://img.shields.io/badge/Instagram-ketsar.aaw-E4405F?style=for-the-badge&logo=instagram)](https://www.instagram.com/ketsar.aaw/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-ketsar-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/ketsar)
[![Streamlit](https://img.shields.io/badge/Streamlit-ketsar28-FF4B4B?style=for-the-badge&logo=streamlit)](https://share.streamlit.io/user/ketsar28)
[![WhatsApp](https://img.shields.io/badge/WhatsApp-Contact-25D366?style=for-the-badge&logo=whatsapp)](https://api.whatsapp.com/send/?phone=6285155343380&text=Halo%20Ketsar!%20Saya%20tertarik%20dengan%20project%20Face%20Analysis%20Anda)

</div>

### 💼 Professional Links

- 📧 **Email**: Available on LinkedIn
- 🌟 **Portfolio**: Check out my other projects on GitHub
- 🤗 **AI Models**: Explore my models on HuggingFace
- 🎨 **Web Apps**: Try my apps on Streamlit

---

## ⭐ Show Your Support

Jika project ini membantu Anda atau Anda merasa terkesan, jangan lupa berikan ⭐ pada repository ini!

---

## 📊 Project Stats

![GitHub Stars](https://img.shields.io/github/stars/ketsar28/real-time-facial-recognition-app?style=social)
![GitHub Forks](https://img.shields.io/github/forks/ketsar28/real-time-facial-recognition-app?style=social)
![GitHub Watchers](https://img.shields.io/github/watchers/ketsar28/real-time-facial-recognition-app?style=social)

---

## 🎓 Academic & Research Use

Jika Anda menggunakan project ini untuk penelitian atau akademik, mohon cantumkan referensi:

```bibtex
@software{ketsar_face_analysis_2025,
  author = {Ketsar Ali},
  title = {Real-Time Face Analysis Web Application},
  year = {2025},
  url = {https://github.com/ketsar28/real-time-facial-recognition-app}
}
```

---

<div align="center">

### 🌟 Thank You for Visiting! 🌟

**Made with 💻 and ☕ by Ketsar Ali**

---

**Copyright © 2025 Ketsar Ali. All Rights Reserved.**

*This project and all associated code, documentation, and materials are the intellectual property of Ketsar Ali.*

---

[⬆ Back to Top](#-real-time-face-analysis-web-application)

</div>
