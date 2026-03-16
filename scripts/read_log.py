import os

def read_log(path):
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return
    
    try:
        # Try different encodings
        for enc in ['utf-8', 'utf-16', 'utf-16-le', 'cp1252']:
            try:
                with open(path, 'r', encoding=enc) as f:
                    content = f.read()
                    print(f"--- LOG START ({enc}) ---")
                    print(content)
                    print("--- LOG END ---")
                    return
            except Exception:
                continue
    except Exception as e:
        print(f"Read Error: {e}")

if __name__ == "__main__":
    read_log("test_final.log")
