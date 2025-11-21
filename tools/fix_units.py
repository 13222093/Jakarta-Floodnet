import pandas as pd
import os

FILE_PATH = 'data/DATASET_FINAL_TRAINING.csv'

def fix_units():
    print(f"🔧 Fixing units in {FILE_PATH}...")
    
    if not os.path.exists(FILE_PATH):
        print(f"❌ File not found: {FILE_PATH}")
        return

    try:
        # Load data
        df = pd.read_csv(FILE_PATH)
        
        # Check if column exists
        if 'tma_manggarai' not in df.columns:
            print("❌ Column 'tma_manggarai' not found!")
            print(df.columns)
            return

        # Divide by 10
        print("   Dividing 'tma_manggarai' by 10...")
        df['tma_manggarai'] = df['tma_manggarai'] / 10.0
        
        # Save back to CSV
        df.to_csv(FILE_PATH, index=False)
        print("✅ File overwritten successfully.")
        
        # Verify
        print("\n👀 First 5 rows (Verification):")
        print(df.head())
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    fix_units()
