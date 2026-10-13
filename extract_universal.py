"""
Универсальный экстрактор текста из PDF.

Режимы:
    python extract_universal.py                # обработать ВСЕ PDF в data/ (перезаписать pages.json)
    python extract_universal.py --add <pdf>    # добавить ОДИН PDF к существующему pages.json

"""

import argparse
import io
import json
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# Настройки OCR
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Users\l.ilyintseva\Downloads\tesseract-ocr-w64-setup-5.5.3.20260724\tesseract.exe"
)
OCR_LANG = "rus+eng"
OCR_CONFIG = "--psm 6"
RENDER_DPI = 300
MIN_CHARS_FOR_TEXT_LAYER = 500

PAGES_PATH = Path("pages.json")


def has_text_layer(pdf_path):
    doc = pymupdf.open(pdf_path)
    total = sum(len(page.get_text().strip()) for page in doc)
    doc.close()
    return total > MIN_CHARS_FOR_TEXT_LAYER


def extract_text_layer(pdf_path):
    doc = pymupdf.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        pages.append({
            "page": i + 1,
            "text": text,
            "n_chars": len(text),
            "source": Path(pdf_path).name,
            "method": "text_layer",
        })
    doc.close()
    return pages


def extract_ocr(pdf_path):
    doc = pymupdf.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        zoom = RENDER_DPI / 72
        mat = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(
            img, lang=OCR_LANG, config=OCR_CONFIG
        ).strip()
        pages.append({
            "page": i + 1,
            "text": text,
            "n_chars": len(text),
            "source": Path(pdf_path).name,
            "method": "ocr",
        })
        print(f"   стр. {i+1}: {len(text)} симв. (OCR)")
    doc.close()
    return pages


def extract(pdf_path, force_ocr=False):
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Файл не найден: {pdf_path}")

    print(f"\nФайл: {pdf_path.name}")

    if force_ocr or not has_text_layer(pdf_path):
        print("      Текстовый слой отсутствует — запускаю OCR...")
        pages = extract_ocr(pdf_path)
    else:
        print("      Текстовый слой найден — извлекаю напрямую.")
        pages = extract_text_layer(pdf_path)

    total = sum(p["n_chars"] for p in pages)
    print(f"   Всего символов: {total}")
    return pages


def make_doc_entry(pdf_path, pages):
    return {
        "source": Path(pdf_path).name,
        "method": pages[0]["method"] if pages else "unknown",
        "n_pages": len(pages),
        "pages": pages,
    }


# Сценарии
def process_all(data_dir="data", output_path=PAGES_PATH):
    """Первичная обработка: ВСЕ PDF → pages.json (перезапись)."""
    data_dir = Path(data_dir)
    pdfs = sorted(data_dir.glob("*.pdf"))

    if not pdfs:
        print(f" В папке {data_dir} не найдено PDF.")
        return

    print(f" Найдено PDF: {len(pdfs)}")

    result = {}
    for pdf in pdfs:
        pages = extract(pdf)
        result[pdf.name] = make_doc_entry(pdf, pages)

    # Бэкап — если pages.json уже существует
    output_path = Path(output_path)
    if output_path.exists():
        backup_path = output_path.with_name("pages.backup.json")
        with open(output_path, "r", encoding="utf-8") as f_src:
            old_data = json.load(f_src)
        with open(backup_path, "w", encoding="utf-8") as f_dst:
            json.dump(old_data, f_dst, ensure_ascii=False, indent=2)
        print(f"\n Бэкап: {backup_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n Сохранено в {output_path}")
    for name, data in result.items():
        print(f"   • {name}: {data['n_pages']} стр., метод: {data['method']}")


def add_one(pdf_path, output_path=PAGES_PATH):
    """Добавить ОДИН PDF к существующему pages.json."""
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)
    doc_name = pdf_path.name

    if not pdf_path.exists():
        print(f" Файл не найден: {pdf_path}")
        return

    # --- Загружаем существующий pages.json ---
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f" Загружено из {output_path}: {len(data)} документов")

        # --- Бэкап ДО изменений ---
        backup_path = output_path.with_name("pages.backup.json")
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f" Бэкап: {backup_path}")
    else:
        data = {}
        print(f" {output_path} не найден — будет создан")

    # Проверяем, есть ли уже
    if doc_name in data:
        print(f"\n Документ '{doc_name}' уже есть в pages.json.")
        print(f"   Пропускаю. Удалите его вручную, если нужно переделать.")
        return

    # Обрабатываем и добавляем
    pages = extract(pdf_path)
    data[doc_name] = make_doc_entry(pdf_path, pages)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n Добавлено в {output_path}")
    print(f"   Всего документов: {len(data)}")
    for name, d in data.items():
        print(f"   • {name}: {d['n_pages']} стр., метод: {d['method']}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Универсальный экстрактор PDF")
    parser.add_argument(
        "--add", type=str, default=None,
        help="Добавить ОДИН PDF к существующему pages.json"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Папка с PDF (для режима без аргументов)"
    )
    args = parser.parse_args()

    if args.add:
        add_one(args.add, output_path=PAGES_PATH)
    else:
        process_all(data_dir=args.data_dir, output_path=PAGES_PATH)