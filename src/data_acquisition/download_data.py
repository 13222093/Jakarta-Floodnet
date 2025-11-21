import pandas as pd
import os

# Buat folder data kalau belum ada
if not os.path.exists('data'):
    os.makedirs('data')

print("⏳ Sedang mendownload data Hujan Katulampa (2022-2024)...")

# URL API Open-Meteo
url = "https://archive-api.open-meteo.com/v1/archive?latitude=-6.6332&longitude=106.8368&start_date=2022-01-01&end_date=2024-01-01&hourly=precipitation&timezone=Asia%2FBangkok&format=csv"

try:
    # Baca CSV langsung dari URL (Skip 3 baris pertama karena itu header metadata)
    df = pd.read_csv(url, skiprows=3)
    
    # Simpan ke folder data
    output_path = 'data/hujan_bogor.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✅ BERHASIL! Data tersimpan di: {output_path}")
    print(df.head()) # Tampilkan 5 baris pertama buat ngecek
    
except Exception as e:
    print(f"❌ GAGAL: {e}")