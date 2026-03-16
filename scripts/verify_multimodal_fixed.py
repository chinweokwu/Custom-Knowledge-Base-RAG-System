import sys
import os
try:
    import pytesseract
    print(f"pytesseract found. Library version: {pytesseract.__version__}")
except Exception as e:
    print(f"pytesseract error: {e}")

try:
    import unstructured
    print(f"unstructured version: {unstructured.__version__}")
except ImportError:
    print("unstructured NOT FOUND")

# Check for Tesseract Binary
tess_paths = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Users\mwx1328172\AppData\Local\Tesseract-OCR\tesseract.exe",
    r"D:\Tesseract-OCR\tesseract.exe"
]
for p in tess_paths:
    if os.path.exists(p):
        print(f"Tesseract Binary Found at: {p}")
        break
else:
    print("Tesseract Binary NOT FOUND in common locations.")
