import os
import ctypes

def get_dir_size(start_path):
    total_size = 0
    try:
        if not os.path.exists(start_path) or not os.path.isdir(start_path):
            return 0
        for dirpath, dirnames, filenames in os.walk(start_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    try:
                        total_size += os.path.getsize(fp)
                    except OSError:
                        pass
    except Exception:
        pass
    return total_size

def get_top_large_folders(root_dir, depth=2):
    sizes = {}
    try:
        # Just scan top level folders inside the root_dir
        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            if os.path.isdir(item_path):
                # skip Windows restricted folders typically
                if item.lower() in ["windows", "program files", "program files (x86)"]:
                    continue
                size = get_dir_size(item_path)
                sizes[item_path] = size
    except Exception as e:
        print(f"Error accessing {root_dir}: {e}")
        
    return sorted(sizes.items(), key=lambda x: x[1], reverse=True)[:10]

print("Scanning Users directory...")
user_dir = r"C:\Users\mwx1328172"
top_user = get_top_large_folders(user_dir)
print("\nTop 10 Largest Folders in", user_dir)
for p, s in top_user:
    print(f"{p}: {s / (1024**3):.2f} GB")

print("\nScanning specific hidden AppData folders...")
appdata_local = os.path.join(user_dir, "AppData", "Local")
top_local = get_top_large_folders(appdata_local)
for p, s in top_local:
    print(f"{p}: {s / (1024**3):.2f} GB")
