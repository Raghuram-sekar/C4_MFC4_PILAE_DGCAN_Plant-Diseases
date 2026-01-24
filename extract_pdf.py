import PyPDF2
import sys

try:
    reader = PyPDF2.PdfReader("MFC-Base Paper.pdf")
    # Print pages 2 to 12 (indices 1 to 11) to capture methodology
    for i in range(1, min(12, len(reader.pages))):
        print(f"\n--- Page {i+1} ---")
        text = reader.pages[i].extract_text()
        print(text)
except Exception as e:
    print(f"Error: {e}")
