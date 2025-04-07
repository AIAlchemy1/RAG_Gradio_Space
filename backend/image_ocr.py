# backend/image_ocr.py

from PIL import Image, ImageOps, ImageFilter
import easyocr
import numpy as np

# ✅ Initialize EasyOCR reader globally (only once)
reader = easyocr.Reader(['en'], verbose=False)

def preprocess_image(img: Image.Image) -> np.ndarray:
    """Preprocess the image for better OCR and convert to numpy array."""
    img = img.convert("L")  # Convert to grayscale
    img = ImageOps.autocontrast(img)  # Improve contrast
    img = img.filter(ImageFilter.SHARPEN)  # Sharpen
    return np.array(img)  # Return as numpy array for EasyOCR

def clean_ocr_text(text: str) -> str:
    """Post-process OCR text to remove short noisy lines."""
    lines = text.splitlines()
    cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 5]
    return " ".join(cleaned_lines)

def extract_text_from_image(image: Image.Image) -> str:
    """Extract and clean text from a preprocessed PIL image."""
    try:
        img_array = preprocess_image(image)

        # OCR
        results = reader.readtext(img_array)

        # Extract only the text parts
        extracted_text = " ".join([text for _, text, _ in results])

        if extracted_text.strip():
            cleaned_text = clean_ocr_text(extracted_text)
            return cleaned_text or "⚠️ OCR ran but no meaningful text found."
        else:
            return "⚠️ OCR ran but no readable text found."

    except Exception as e:
        return f"🚫 OCR runtime error: {e}"
