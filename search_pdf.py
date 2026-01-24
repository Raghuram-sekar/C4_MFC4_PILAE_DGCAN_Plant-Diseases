import PyPDF2

keywords = ["color", "grayscale", "segmented", "RGB", "channels", "PlantVillage"]

try:
    reader = PyPDF2.PdfReader("MFC-Base Paper.pdf")
    print(f"Total Pages: {len(reader.pages)}")
    
    found_sentences = []
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text: continue
        
        # Simple sentence splitting
        sentences = text.replace('\n', ' ').split('.')
        
        for sentence in sentences:
            for kw in keywords:
                if kw.lower() in sentence.lower():
                    found_sentences.append(f"[Page {i+1}] {sentence.strip()}")
                    break
    
    # Print unique sentences to avoid duplicates
    for s in list(set(found_sentences)):
        print(s)
        print("-" * 20)
        
except Exception as e:
    print(f"Error: {e}")
