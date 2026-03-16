import pandas as pd
import os

def analyze_excel(file_path):
    try:
        if not os.path.exists(file_path):
            print(f"ERROR: File not found at {file_path}")
            return
        
        # Using openpyxl as engine for Excel logs
        df = pd.read_excel(file_path, engine='openpyxl')
        print(f"EXCEL_STRUCTURE_REPORT: {os.path.basename(file_path)}")
        print(f"COLUMNS: {list(df.columns)}")
        print(f"TOTAL_ROWS: {len(df)}")
        
        print("\n--- FIRST 3 ROWS SAMPLE ---")
        print(df.head(3).to_string())
        
        # Specifically look for the target ID
        found = False
        for col in df.columns:
            matches = df[df[col].astype(str).str.contains("jwx1369347", case=False, na=False)]
            if not matches.empty:
                print(f"\n--- MATCH FOUND IN COLUMN: {col} ---")
                print(matches.head(1).to_string())
                found = True
        
        if not found:
            print("\nWARNING: Target ID 'jwx1369347' not found in any column of the original file!")

    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    file_path = r"D:\AI knowledge Based\uploads\rnoc_llm_c_chat_log_export_20260309164536.xlsx"
    analyze_excel(file_path)
