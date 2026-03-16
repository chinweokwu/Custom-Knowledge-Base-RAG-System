import os
import sys
from pathlib import Path

paths_of_interest = [
    'TEMP', 'TMP', 'LOCALAPPDATA', 'APPDATA', 'USERPROFILE', 
    'HOME', 'PYTHONPATH', 'PYTHONUSERBASE', 'PIP_CACHE_DIR',
    'HF_HOME', 'XDG_CACHE_HOME', 'NLTK_DATA'
]

print("--- Environment Paths Audit ---")
for env in paths_of_interest:
    val = os.environ.get(env)
    print(f"{env}: {val}")

print("\n--- Python System Paths ---")
for p in sys.path:
    print(p)

print("\n--- Project Directory ---")
print(Path.cwd())
