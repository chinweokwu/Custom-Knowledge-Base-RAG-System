import os

def get_dir_size(start_path):
    total_size = 0
    try:
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

user_dir = r"C:\Users\mwx1328172"
print(f"Scanning directories in {user_dir}...")
sizes = {}

for item in os.listdir(user_dir):
    item_path = os.path.join(user_dir, item)
    if os.path.isdir(item_path):
        sizes[item] = get_dir_size(item_path)

print("\n--- Top Folders in User Directory ---")
for folder, size in sorted(sizes.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{folder}: {size / (1024**3):.2f} GB")

# Also scan AppData\Local specifically
appdata_local = os.path.join(user_dir, "AppData", "Local")
local_sizes = {}
if os.path.exists(appdata_local):
    print(f"\nScanning directories in {appdata_local}...")
    for item in os.listdir(appdata_local):
        item_path = os.path.join(appdata_local, item)
        if os.path.isdir(item_path):
            local_sizes[item] = get_dir_size(item_path)

    print("\n--- Top Folders in AppData\\Local ---")
    for folder, size in sorted(local_sizes.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{folder}: {size / (1024**3):.2f} GB")
