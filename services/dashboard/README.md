# 🌊 Jakarta FloodNet Dashboard

Dashboard interaktif berbasis Streamlit untuk sistem monitoring dan prediksi banjir Jakarta FloodNet.

## 📋 Deskripsi

Dashboard ini merupakan **frontend client** yang mengonsumsi API dari Backend Gateway (FastAPI). Dashboard **tidak memuat model ML secara langsung**, melainkan berkomunikasi dengan backend untuk mendapatkan prediksi dan analisis.

## 🏗️ Arsitektur

```
Dashboard (Streamlit) ──HTTP Requests──> API Gateway (FastAPI) ──> ML Models (LSTM + YOLO)
                                                                      ├─ LSTM: Prediksi
                                                                      └─ YOLO: Verifikasi Visual
```

## 🎯 Fitur Utama

### 1. 📊 Dashboard Utama
- **Health Check**: Status koneksi ke API Gateway
- **System Monitoring**: Status semua komponen sistem (LSTM, YOLO, API)
- **Real-time Metrics**: Ketinggian air, tren, dan grafik historis
- **Auto Refresh**: Pembaruan data otomatis setiap 30 detik

### 2. 🔮 Prediksi Banjir (LSTM)
- **Input Form**: Curah hujan Bogor dan Jakarta
- **Visual Indicators**: Color-coded warning levels
- **Prediction Results**: Level air, risiko, confidence, rekomendasi
- **Interactive Charts**: Perbandingan curah hujan

### 3. 👁️ Verifikasi Visual (YOLO)
- **Image Upload**: Support JPG, PNG (max 10MB)
- **Object Detection**: Deteksi objek terkait banjir
- **Analysis Results**: Flood indicators, probability, severity
- **Detail View**: Tabel deteksi objek dengan confidence scores

## 📁 Struktur Folder

```
services/dashboard/
├── src/
│   ├── app.py                          # Entry point utama
│   ├── components/
│   │   ├── __init__.py
│   │   ├── api_client.py               # Client untuk API Gateway
│   │   └── ui_widgets.py               # UI components reusable
│   └── pages/
│       ├── 01_Dashboard_Utama.py       # Halaman monitoring
│       ├── 02_Prediksi_Banjir.py       # Halaman prediksi LSTM
│       └── 03_Verifikasi_Visual.py     # Halaman analisis YOLO
├── requirements.txt                     # Dependencies
└── README.md                           # Dokumentasi ini
```

## 🚀 Installation

### 1. Install Dependencies

```bash
cd services/dashboard
pip install -r requirements.txt
```

Atau menggunakan `uv`:

```bash
uv pip install -r requirements.txt
```

### 2. Pastikan API Gateway Berjalan

Dashboard memerlukan API Gateway yang berjalan di `http://localhost:8000`.

```bash
# Di terminal terpisah, jalankan API Gateway
cd services/api_gateway/src
python main.py
```

### 3. Jalankan Dashboard

```bash
cd services/dashboard/src
streamlit run app.py
```

Dashboard akan terbuka di browser pada `http://localhost:8501`.

## ⚙️ Konfigurasi

### API Base URL

Default: `http://localhost:8000`

Untuk mengubah, edit file `components/api_client.py`:

```python
API_BASE_URL = "http://your-api-server:8000"
```

### Timeouts

Default timeout untuk API requests: 30 seconds

Ubah di `api_client.py`:

```python
self.timeout = 60  # seconds
```

## 🎨 Tampilan & UX

### Color Scheme

- **🟢 Hijau (Safe)**: Kondisi aman, tidak ada risiko
- **🟡 Kuning (Warning)**: Siaga, perlu perhatian
- **🔴 Merah (Danger)**: Bahaya, evakuasi diperlukan
- **🔵 Biru (Info)**: Informasi umum

### Responsive Design

- **Wide Mode**: Optimal untuk desktop (1920x1080)
- **Adaptive Layout**: Menyesuaikan dengan ukuran layar
- **Mobile Friendly**: Tetap usable di tablet/mobile

## 📊 Cara Penggunaan

### Dashboard Utama

1. Buka halaman utama (`http://localhost:8501`)
2. Check status koneksi di sidebar
3. Lihat metrics real-time dan grafik tren
4. Gunakan toggle "Auto Refresh" untuk update otomatis
5. Klik tombol aksi cepat untuk navigasi

### Prediksi Banjir

1. Navigasi ke **"Prediksi Banjir"** dari sidebar
2. Masukkan curah hujan Bogor (mm)
3. Masukkan curah hujan Jakarta (mm)
4. Klik **"Hitung Prediksi"**
5. Lihat hasil: Level air, risiko, dan rekomendasi

### Verifikasi Visual

1. Navigasi ke **"Verifikasi Visual"**
2. Upload gambar kondisi lapangan (JPG/PNG)
3. Preview gambar akan muncul
4. Klik **"Analisis Gambar"**
5. Lihat hasil: Deteksi objek, flood probability, severity

## 🔧 API Endpoints yang Digunakan

Dashboard berkomunikasi dengan endpoints berikut:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Check API availability |
| `/status` | GET | Get system status |
| `/predict` | POST | LSTM flood prediction |
| `/verify-visual` | POST | YOLO image analysis |
| `/history` | GET | Get historical data |

## 🐛 Troubleshooting

### API Connection Error

```
❌ API Gateway tidak dapat dijangkau
```

**Solution:**
1. Pastikan API Gateway berjalan: `cd services/api_gateway/src && python main.py`
2. Check port 8000 tidak digunakan aplikasi lain: `lsof -i :8000`
3. Verifikasi firewall tidak memblokir port

### Model Not Loaded

```
⚠️ LSTM Model: unknown
```

**Solution:**
1. Check API Gateway logs untuk error loading model
2. Pastikan model files ada di `models/` directory
3. Restart API Gateway

### Slow Response

**Solution:**
1. Check CPU/GPU usage pada server
2. Increase timeout di `api_client.py`
3. Optimize image size sebelum upload (< 1MB recommended)

## 📈 Performance Tips

### Optimasi Dashboard

1. **Auto Refresh**: Matikan jika tidak diperlukan (menghemat bandwidth)
2. **Image Upload**: Compress gambar sebelum upload
3. **Caching**: Streamlit automatically caches API responses

### Best Practices

- Gunakan gambar resolusi optimal (640x640 atau 1280x720)
- Batch predictions jika memungkinkan
- Monitor memory usage dengan `st.cache_data` untuk fungsi berat

## 🔐 Security Considerations

### Production Deployment

Untuk deployment production:

1. **HTTPS**: Gunakan SSL/TLS untuk enkripsi
2. **Authentication**: Tambahkan login system
3. **API Keys**: Implement API key authentication
4. **Rate Limiting**: Limit requests per user
5. **Input Validation**: Sanitize semua user inputs

### Environment Variables

```bash
# .env file
API_BASE_URL=https://api.floodnet.id
API_KEY=your-secret-key
STREAMLIT_SERVER_PORT=8501
```

## 🚢 Deployment

### Local Development

```bash
streamlit run src/app.py
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
EXPOSE 8501

CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Cloud Deployment

**Streamlit Cloud:**
```bash
# Push to GitHub, then connect to Streamlit Cloud
# Set secrets in Streamlit Cloud dashboard
```

**Heroku:**
```bash
# Procfile
web: streamlit run src/app.py --server.port=$PORT
```

## 📚 Dependencies

- **streamlit**: Web framework
- **requests**: HTTP client
- **pandas**: Data processing
- **plotly**: Interactive charts
- **Pillow**: Image processing

Lihat `requirements.txt` untuk versi lengkap.

## 🤝 Contributing

1. Fork repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📝 License

Project ini dikembangkan untuk kepentingan mitigasi bencana dan early warning system.

## 👥 Team

**Jakarta FloodNet Team**

---

**Dashboard Version**: 1.0.0  
**Last Updated**: November 2025  
**Framework**: Streamlit 1.39.0  
**Python**: 3.11+
