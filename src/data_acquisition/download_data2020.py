import pandas as pd
import os

# ==========================================
# FILE EXCEL/CSV 2020 KAMU TARUH DISINI
# ==========================================
FILE_TMA_CSV = 'data/tma_2020.xlsx.csv' 
FILE_OUTPUT = 'data/DATASET_FINAL_TRAINING.csv'

print("🚀 BACK TO 2020: MEMBUAT DATASET BANJIR BESAR...")

# 1. DOWNLOAD HUJAN BOGOR (2020)
print("1️⃣  Download Hujan Bogor...")
url_bogor = "https://archive-api.open-meteo.com/v1/archive?latitude=-6.6332&longitude=106.8368&start_date=2020-01-01&end_date=2020-12-31&hourly=precipitation&timezone=Asia%2FBangkok&format=csv"
df_bogor = pd.read_csv(url_bogor, skiprows=3)
df_bogor['time'] = pd.to_datetime(df_bogor['time'])
df_bogor = df_bogor.set_index('time').rename(columns={'precipitation (mm)': 'hujan_bogor'})

# 2. DOWNLOAD HUJAN JAKARTA (2020)
print("2️⃣  Download Hujan Jakarta...")
url_jkt = "https://archive-api.open-meteo.com/v1/archive?latitude=-6.2078&longitude=106.8485&start_date=2020-01-01&end_date=2020-12-31&hourly=precipitation&timezone=Asia%2FBangkok&format=csv"
df_jkt = pd.read_csv(url_jkt, skiprows=3)
df_jkt['time'] = pd.to_datetime(df_jkt['time'])
df_jkt = df_jkt.set_index('time').rename(columns={'precipitation (mm)': 'hujan_jakarta'})

# 3. PROSES CSV TMA (Yang kamu punya)
print("3️⃣  Proses CSV TMA 2020...")
try:
    # Baca CSV (bukan Excel)
    df_tma = pd.read_csv(FILE_TMA_CSV)
    df_tma.columns = df_tma.columns.str.lower()
    
    # Filter khusus Manggarai
    # Data kamu format panjang (long format), bukan lebar
    # Kolom: nama_pintu_air, lokasi, ..., tanggal, tinggi_air
    
    manggarai_mask = df_tma['nama_pintu_air'].str.contains('Manggarai', case=False, na=False)
    df_manggarai = df_tma[manggarai_mask].copy()
    
    if df_manggarai.empty:
        print("   ❌ Tidak ada data Manggarai ditemukan di CSV!")
    else:
        # Ambil kolom waktu dan nilai
        df_manggarai = df_manggarai[['tanggal', 'tinggi_air']].copy()
        df_manggarai.columns = ['timestamp', 'tma_manggarai']
        
        # Convert datetime
        df_manggarai['timestamp'] = pd.to_datetime(df_manggarai['timestamp'], errors='coerce')
        df_manggarai = df_manggarai.dropna().set_index('timestamp')
        
        # Resample per jam (ambil rata-rata jika ada duplikat jam)
        df_manggarai = df_manggarai.resample('1H').mean().interpolate()
        
        # 4. GABUNGKAN SEMUA
        print("4️⃣  Merging...")
        # Inner join biar aman irisan waktunya
        df_final = df_bogor.join(df_jkt, how='inner').join(df_manggarai, how='inner')
        
        df_final.to_csv(FILE_OUTPUT)
        print(f"✅ DONE! Dataset siap di: {FILE_OUTPUT}")
        print(df_final.head())
        print(f"Total baris: {len(df_final)}")

except Exception as e:
    print(f"❌ Error proses data: {e}")