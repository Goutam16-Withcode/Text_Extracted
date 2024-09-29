import easyocr
import json

def extract_text(image_path):
    reader = easyocr.Reader(['en', 'hi'])  
    results = reader.readtext(image_path)

    extracted_text = []
    for (bbox, text, prob) in results:
        extracted_text.append(text)

    return json.dumps({"extracted_text": " ".join(extracted_text)})
