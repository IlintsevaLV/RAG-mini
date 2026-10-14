"""
Извлечение текста из PDF с разделением на «чистый текст» и «таблицы».

Три ветки обработки:
    1. OCR          — для сканов (нет текстового слоя). Tesseract.
    2. Hybrid       — для текстовых PDF: PyMuPDF (текст + BBox) +
                      Docling (только таблицы) + пространственное слияние.
                      Плюс «умный caption»: заголовок таблицы ищется в
                      соседнем блоке PyMuPDF над таблицей.
    3. Text layer   — fallback, если Docling отключён или упал.

Режимы CLI:
    python extract.py                # обработать ВСЕ PDF в data/ (перезаписать pages.json)
    python extract.py --add <pdf>    # добавить ОДИН PDF к существующему pages.json

Tesseract ожидается в PROJECT_ROOT/models/tesseract/.
"""

import argparse
import io
import json
import re
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image
import os

Image.MAX_IMAGE_PIXELS = None

# ПУТИ ПРОЕКТА

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
PAGES_PATH = BASE_DIR / "pages.json"

TESSERACT_DIR = PROJECT_ROOT / "models" / "tesseract"
TESSERACT_EXE = TESSERACT_DIR / "tesseract.exe"
TESSDATA_DIR = TESSERACT_DIR / "tessdata"


# НАСТРОЙКИ OCR

OCR_LANG = "rus+eng"
OCR_CONFIG = "--psm 6"
RENDER_DPI = 300
MIN_CHARS_FOR_TEXT_LAYER = 500

if not TESSERACT_EXE.exists():
    raise FileNotFoundError(
        f"Tesseract не найден: {TESSERACT_EXE}\n"
        f"Скопируйте папку с tesseract.exe (и DLL) в {TESSERACT_DIR}"
    )

pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_EXE)
os.environ["TESSDATA_PREFIX"] = str(TESSDATA_DIR)

# НАСТРОЙКИ ГИБРИДНОГО РЕЖИМА

USE_DOCLING = True                  # глобальный переключатель гибрида
BBOX_OVERLAP_THRESHOLD = 0.3        # доля блока, попадающая в таблицу, чтобы отбросить

# Параметры «умного caption»
CAPTION_MAX_DISTANCE_PT = 45.0      # макс. расстояние по Y от блока до таблицы
CAPTION_KEYWORDS = re.compile(
    r"(таблиц|продолж|table|табл\.)",
    re.IGNORECASE,
)

# ВЕТКА 1: OCR (для сканов)
def has_text_layer(pdf_path):
    """True, если в PDF есть настоящий текстовый слой (не скан)."""
    doc = pymupdf.open(pdf_path)
    total = sum(len(page.get_text().strip()) for page in doc)
    doc.close()
    return total > MIN_CHARS_FOR_TEXT_LAYER


def extract_ocr(pdf_path):
    """OCR страниц через Tesseract. Возвращает список страниц."""
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
            "tables": [],
            "n_chars": len(text),
            "n_tables": 0,
            "source": Path(pdf_path).name,
            "method": "ocr",
        })
        print(f"   стр. {i+1}: {len(text)} симв. (OCR)")
    doc.close()
    return pages


# ВЕТКА 2: Чистый PyMuPDF (fallback, если Docling недоступен)

def extract_text_layer(pdf_path):
    """Простое извлечение текстового слоя через PyMuPDF."""
    doc = pymupdf.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        pages.append({
            "page": i + 1,
            "text": text,
            "tables": [],
            "n_chars": len(text),
            "n_tables": 0,
            "source": Path(pdf_path).name,
            "method": "text_layer",
        })
    doc.close()
    return pages



# ВЕТКА 3: ГИБРИД (PyMuPDF + Docling)

def bbox_intersection_area(a, b):
    """Площадь пересечения двух прямоугольников (x0, y0, x1, y1)."""
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def bbox_area(b):
    """Площадь прямоугольника."""
    return max(0.0, (b[2] - b[0]) * (b[3] - b[1]))


def overlaps(a, b, threshold=BBOX_OVERLAP_THRESHOLD):
    """True, если блок a существенно перекрывается с b.

    Сравниваем площадь пересечения с площадью a.
    """
    inter = bbox_intersection_area(a, b)
    if inter == 0.0:
        return False
    return inter / max(bbox_area(a), 1e-6) > threshold


def overlaps_horizontally(a, b, min_frac=0.2):
    """True, если блоки a и b перекрываются по X хотя бы на min_frac от ширины a."""
    x0 = max(a[0], b[0])
    x1 = min(a[2], b[2])
    if x1 <= x0:
        return False
    width_a = max(a[2] - a[0], 1e-6)
    return (x1 - x0) / width_a >= min_frac


def extract_text_blocks_with_bbox(pdf_path):
    """PyMuPDF: страницы → список текстовых блоков с BBox (TOPLEFT, points)."""
    doc = pymupdf.open(pdf_path)
    result = []
    for page in doc:
        raw = page.get_text("blocks")
        page_blocks = []
        for b in raw:
            x0, y0, x1, y1, text, block_no, block_type = b
            if block_type != 0:
                continue
            text = text.strip()
            if not text:
                continue
            page_blocks.append({
                "text": text,
                "bbox": (float(x0), float(y0), float(x1), float(y1)),
            })
        result.append(page_blocks)
    doc.close()
    return result


def extract_tables_with_docling(pdf_path, page_heights):
    """Docling: только таблицы. BBox приводится к TOPLEFT (points).

    Args:
        pdf_path: путь к PDF
        page_heights: список высот страниц в points (из PyMuPDF),
                      нужен для нормализации BOTTOMLEFT → TOPLEFT.

    Returns:
        Список по страницам, где каждая страница — список словарей:
            {"bbox": (x0, y0, x1, y1), "markdown": str, "caption": str, "label": str}
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.do_ocr = False                # не трогаем обычный текст
    pipeline_options.table_structure_options.do_cell_matching = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    result = converter.convert(str(pdf_path))

    pages_tables = [[] for _ in range(len(page_heights))]

    for table in result.document.tables:
        # Фильтр по label: только реальные таблицы (не оглавления)
        label_str = str(getattr(table, "label", "")).lower()
        if "table" not in label_str or "document_index" in label_str:
            continue

        if not table.prov:
            continue
        prov = table.prov[0]
        page_no = prov.page_no
        if not (1 <= page_no <= len(page_heights)):
            continue

        page_height = page_heights[page_no - 1]

        # --- Нормализация BBox к TOPLEFT ---
        bbox = None
        if prov.bbox is not None:
            bb = prov.bbox
            if bb.coord_origin.name == "BOTTOMLEFT":
                y0 = page_height - float(bb.b)
                y1 = page_height - float(bb.t)
            else:
                y0 = float(bb.t)
                y1 = float(bb.b)
            # Страховка от перепутанных t/b
            if y0 > y1:
                y0, y1 = y1, y0
            bbox = (float(bb.l), y0, float(bb.r), y1)

        # --- Markdown ---
        try:
            markdown = table.export_to_markdown(doc=result.document)
        except TypeError:
            markdown = table.export_to_markdown()

        # --- Caption от Docling (если он его сам нашёл) ---
        caption = ""
        try:
            cap_list = getattr(table, "captions", []) or []
            parts = []
            for ref in cap_list:
                cref = getattr(ref, "cref", None)
                if not cref:
                    continue
                try:
                    _, _, idx = cref.split("/")
                    item = result.document.texts[int(idx)]
                    parts.append(str(item.text))
                except Exception:
                    pass
            caption = " ".join(parts).strip()
        except Exception:
            pass

        pages_tables[page_no - 1].append({
            "bbox": bbox,
            "markdown": markdown,
            "caption": caption,
            "label": label_str,
        })

    return pages_tables


def _find_smart_caption(table_bbox, blocks, used_block_ids):
    """Ищет «умный caption» для таблицы среди блоков PyMuPDF.

    Логика:
        - берём блоки, лежащие НАД таблицей (blk.y1 <= table.y0)
        - на расстоянии не больше CAPTION_MAX_DISTANCE_PT
        - пересекающиеся с таблицей по X
        - из них — ближайший по Y
        - если в тексте есть ключевые слова (таблиц/продолж/table) — это caption

    Returns:
        (text, block_index) или (None, None).
    """
    if table_bbox is None:
        return None, None

    candidates = []
    for i, blk in enumerate(blocks):
        if i in used_block_ids:
            continue
        b = blk["bbox"]
        # Блок выше таблицы
        if b[3] > table_bbox[1]:
            continue
        # Не дальше CAPTION_MAX_DISTANCE_PT
        if table_bbox[1] - b[3] > CAPTION_MAX_DISTANCE_PT:
            continue
        # Пересекается по X
        if not overlaps_horizontally(b, table_bbox):
            continue
        candidates.append((i, blk))

    if not candidates:
        return None, None

    # Ближайший сверху
    idx, blk = max(candidates, key=lambda x: x[1]["bbox"][3])
    text = blk["text"].strip()

    # Проверяем, что это похоже на заголовок таблицы
    if CAPTION_KEYWORDS.search(text):
        return text, idx
    return None, None


def extract_hybrid(pdf_path):
    """ГИБРИД: PyMuPDF (текст + BBox) + Docling (таблицы) + умный caption."""
    pdf_path = Path(pdf_path)
    print("      Гибридный режим: PyMuPDF (текст) + Docling (таблицы)...")

    # Высоты страниц нужны для нормализации BBox Docling
    doc = pymupdf.open(pdf_path)
    page_heights = [p.rect.height for p in doc]
    doc.close()

    # Шаг 1: текстовые блоки PyMuPDF
    pymupdf_pages = extract_text_blocks_with_bbox(pdf_path)
    print(f"      PyMuPDF: {sum(len(p) for p in pymupdf_pages)} блоков")

    # Шаг 2: таблицы Docling
    docling_pages = extract_tables_with_docling(pdf_path, page_heights)
    n_tables = sum(len(p) for p in docling_pages)
    print(f"      Docling: {n_tables} таблиц")

    # Шаг 3: слияние постранично
    pages = []
    for i, (blocks, tables) in enumerate(zip(pymupdf_pages, docling_pages)):
        used_block_ids = set()   # блоки, ушедшие в caption или отброшенные как часть таблицы

        # 3a. Отбрасываем блоки, которые внутри таблиц
        kept_blocks = []
        for idx, blk in enumerate(blocks):
            in_table = False
            for tbl in tables:
                if tbl["bbox"] is None:
                    continue
                if overlaps(blk["bbox"], tbl["bbox"]):
                    in_table = True
                    break
            if in_table:
                used_block_ids.add(idx)
            else:
                kept_blocks.append((idx, blk))

        # 3b. Умный caption — для каждой таблицы без своего caption
        for tbl in tables:
            if tbl["caption"]:
                continue    # Docling уже дал caption
            cap_text, cap_idx = _find_smart_caption(tbl["bbox"], blocks, used_block_ids)
            if cap_text:
                tbl["caption"] = cap_text
                used_block_ids.add(cap_idx)

        # 3c. Убираем из kept_blocks те, что стали caption
        kept_blocks = [(idx, blk) for (idx, blk) in kept_blocks
                       if idx not in used_block_ids]

        # 3d. Сортируем оставшиеся блоки в порядке чтения (Y, потом X)
        kept_blocks.sort(key=lambda x: (round(x[1]["bbox"][1], 1), x[1]["bbox"][0]))

        # 3e. Формируем текст страницы
        text_parts = [blk["text"] for _, blk in kept_blocks]
        text = "\n\n".join(text_parts).strip()

        # 3f. Формируем представление таблиц для чанкинга
        tables_out = []
        for tbl in tables:
            tbl_text = tbl["markdown"]
            if tbl["caption"]:
                tbl_text = f"{tbl['caption']}\n\n{tbl_text}"
            tables_out.append({
                "bbox": tbl["bbox"],
                "markdown": tbl["markdown"],
                "caption": tbl["caption"],
                "text": tbl_text.strip(),
            })

        pages.append({
            "page": i + 1,
            "text": text,
            "tables": tables_out,
            "n_chars": len(text),
            "n_tables": len(tables_out),
            "source": pdf_path.name,
            "method": "hybrid",
        })
        cap_info = f", captions: {sum(1 for t in tables_out if t['caption'])}" if tables_out else ""
        print(f"   стр. {i+1}: {len(text)} симв., {len(tables_out)} табл.{cap_info} (hybrid)")

    return pages


# ДИСПЕТЧЕР
def extract(pdf_path, force_ocr=False):
    """Выбор ветки: OCR / hybrid / text_layer."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Файл не найден: {pdf_path}")

    print(f"\nФайл: {pdf_path.name}")

    if force_ocr or not has_text_layer(pdf_path):
        print("      Текстовый слой отсутствует — запускаю OCR...")
        pages = extract_ocr(pdf_path)
    elif USE_DOCLING:
        try:
            pages = extract_hybrid(pdf_path)
        except Exception as e:
            print(f"      ⚠ Гибридный режим упал ({e}) — fallback на PyMuPDF.")
            pages = extract_text_layer(pdf_path)
    else:
        print("      Текстовый слой найден — извлекаю напрямую (PyMuPDF).")
        pages = extract_text_layer(pdf_path)

    total = sum(p["n_chars"] for p in pages)
    print(f"   Всего символов: {total}")
    return pages

# СОХРАНЕНИЕ
def make_doc_entry(pdf_path, pages):
    """Запись документа для pages.json."""
    return {
        "source": Path(pdf_path).name,
        "method": pages[0]["method"] if pages else "unknown",
        "n_pages": len(pages),
        "pages": pages,
    }


def process_all(data_dir=None, output_path=PAGES_PATH):
    """Первичная обработка: все PDF → pages.json (перезапись + бэкап)."""
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    pdfs = sorted(data_dir.glob("*.pdf"))

    if not pdfs:
        print(f" В папке {data_dir} не найдено PDF.")
        return

    print(f" Найдено PDF: {len(pdfs)}")

    result = {}
    for pdf in pdfs:
        pages = extract(pdf)
        result[pdf.name] = make_doc_entry(pdf, pages)

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
    """Добавить один PDF к существующему pages.json."""
    pdf_path = Path(pdf_path)
    if not pdf_path.is_absolute() and not pdf_path.exists():
        candidate = DATA_DIR / pdf_path.name
        if candidate.exists():
            pdf_path = candidate
    output_path = Path(output_path)
    doc_name = pdf_path.name

    if not pdf_path.exists():
        print(f" Файл не найден: {pdf_path}")
        return

    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f" Загружено из {output_path}: {len(data)} документов")

        backup_path = output_path.with_name("pages.backup.json")
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f" Бэкап: {backup_path}")
    else:
        data = {}
        print(f" {output_path} не найден — будет создан")

    if doc_name in data:
        print(f"\n Документ '{doc_name}' уже есть в pages.json.")
        print(f"   Пропускаю. Удалите его вручную, если нужно переделать.")
        return

    pages = extract(pdf_path)
    data[doc_name] = make_doc_entry(pdf_path, pages)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n Добавлено в {output_path}")
    print(f"   Всего документов: {len(data)}")
    for name, d in data.items():
        print(f"   • {name}: {d['n_pages']} стр., метод: {d['method']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Универсальный экстрактор PDF (extract.py)")
    parser.add_argument(
        "--add", type=str, default=None,
        help="Добавить ОДИН PDF к существующему pages.json"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Папка с PDF (по умолчанию: PROJECT_ROOT/data)"
    )
    args = parser.parse_args()

    if args.add:
        add_one(args.add, output_path=PAGES_PATH)
    else:
        process_all(data_dir=args.data_dir, output_path=PAGES_PATH)