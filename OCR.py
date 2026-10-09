import pymupdf
import pytesseract
from PIL import Image
import io
import json
from pathlib import Path

# --- Отключаем предупреждение PIL про большие картинки ---
Image.MAX_IMAGE_PIXELS = None

# --- Настройки ---
PDF_PATH = Path(r"C:/Users/l.ilyintseva/Desktop/RAG-mini/data/Руководство 23-29.605 по МОС образцов ВС.pdf")
OUTPUT_PATH = Path("ocr_pages_v2.json")

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\l.ilyintseva\Downloads\tesseract-ocr-w64-setup-5.5.3.20260724\tesseract.exe"

OCR_LANG = "rus+eng"
OCR_CONFIG = "--psm 6"     # uniform block of text — лучший результат на тесте
RENDER_DPI = 300


def extract_text_with_ocr(pdf_path):
    doc = pymupdf.open(pdf_path)
    pages_text = []
    for i, page in enumerate(doc):
        zoom = RENDER_DPI / 72
        mat = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img, lang=OCR_LANG, config=OCR_CONFIG)
        text = text.strip()
        pages_text.append({
            "page": i + 1,
            "text": text,
            "n_chars": len(text),
        })
        print(f"  Страница {i+1} обработана: {len(text)} символов")
    doc.close()
    return pages_text


if __name__ == "__main__":
    if not PDF_PATH.exists():
        print(f"Файл не найден: {PDF_PATH}")
        raise SystemExit(1)

    print(f"OCR: lang={OCR_LANG}, config='{OCR_CONFIG}', dpi={RENDER_DPI}")
    pages = extract_text_with_ocr(PDF_PATH)

    total = sum(p["n_chars"] for p in pages)
    print(f"\nГотово! Всего символов: {total}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    print(f"Сохранено в {OUTPUT_PATH}")