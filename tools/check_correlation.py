import pandas as pd
import matplotlib.pyplot as plt
import os

# Define paths
DATA_PATH = 'data/DATASET_FINAL_TRAINING.csv'
OUTPUT_IMAGE = 'sanity_check.png'

def main():
    print("📊 Starting Sanity Check...")
    
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: File not found at {DATA_PATH}")
        return

    try:
        df = pd.read_csv(DATA_PATH)
        # Ensure timestamp is datetime
        # The CSV might have 'timestamp' or 'time' as the index or column. 
        # Based on previous steps, it likely has 'Unnamed: 0' as index if not saved with index=False, 
        # or 'timestamp' if saved properly. Let's inspect columns dynamically or assume 'timestamp' based on previous script.
        # The previous script did: df_final.to_csv(FILE_OUTPUT). 
        # Since it was a join of set_index('time') and set_index('timestamp'), the index name might be mixed or just the index.
        # Let's try to parse the first column as index if it looks like a date, or look for 'timestamp'.
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        elif 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time'])
            df = df.set_index('timestamp')
        else:
            # Fallback: assume first column is the index
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            df = df.set_index(df.columns[0])
            
        print("✅ Data loaded successfully.")
        
        # 2. Filter Data (Jan 1, 2020 - Jan 15, 2020)
        start_date = '2020-01-01'
        end_date = '2020-01-15'
        mask = (df.index >= start_date) & (df.index <= end_date)
        df_sample = df.loc[mask]
        
        if df_sample.empty:
            print("⚠️ Warning: No data found in the specified date range. Plotting first 2 weeks of available data instead.")
            df_sample = df.iloc[:24*14] # First 14 days (assuming hourly)

        print(f"   Plotting {len(df_sample)} data points...")

        # 3. Plotting
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Left Axis (Y1): Hujan Bogor (Bar, Blue, Inverted)
        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Rainfall Bogor (mm)', color=color)
        ax1.bar(df_sample.index, df_sample['hujan_bogor'], color='blue', alpha=0.3, label='Hujan Bogor', width=0.04) # width adjusted for datetime axis
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, df_sample['hujan_bogor'].max() * 3) # Give space
        ax1.invert_yaxis() # Rain falls from top

        # Right Axis (Y2): TMA Manggarai (Line, Red)
        ax2 = ax1.twinx()  
        color = 'tab:red'
        ax2.set_ylabel('TMA Manggarai (cm)', color=color)  
        ax2.plot(df_sample.index, df_sample['tma_manggarai'], color='red', linewidth=2, label='TMA Manggarai')
        ax2.tick_params(axis='y', labelcolor=color)

        # Warning Level Line
        ax2.axhline(y=850, color='orange', linestyle='--', label='Warning Level (850 cm)')

        # Title and Layout
        plt.title('Sanity Check: Bogor Rainfall vs Manggarai Water Level (Jan 2020)')
        fig.tight_layout()  
        
        # Save
        plt.savefig(OUTPUT_IMAGE)
        print(f"✅ Chart saved to {OUTPUT_IMAGE}")

    except Exception as e:
        print(f"❌ Error during plotting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
