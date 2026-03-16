import os
import sys
from langchain_community.document_loaders import UnstructuredPDFLoader
from PIL import Image
import pytesseract

print("--- Multi-Modal Capabilities Audit ---")
print(f"Python Version: {sys.version}")
print(f"Tesseract Path: {getattr(pytesseract, 'pytesseract', {}).get('tesseract_cmd', 'default')}")

try:
    import unstructured
    print("unstructured version:", unstructured.__version__)
except ImportError:
    print("unstructured NOT FOUND")

try:
    img = Image.new('RGB', (60, 30), color = (73, 109, 137))
    print("Pillow (Image handling): OK")
except Exception as e:
    print(f"Pillow Error: {e}")
