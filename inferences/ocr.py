from paddleocr import PaddleOCR

# preload model
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def run_ocr(image_path):
    """
    Run OCR on given image path and return extracted text
    """
    results = ocr.predict(image_path)
    extracted = ""

    if not results or not results[0].get("rec_texts"):
        return ""
    else:
        texts = results[0]["rec_texts"]
        # keeping everyline separate
        extracted = "\n".join(texts)
        return extracted 


if __name__ == "__main__":
    text = run_ocr(r"C:\Users\blott\Desktop\TB2\medguess\saz.png")
    print("Extracted text:\n", text)
