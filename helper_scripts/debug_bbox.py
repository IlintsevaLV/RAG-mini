"""
Отладка BBox: сравнить координаты PyMuPDF и Docling на одной странице.
Автономный скрипт — не зависит от extract_universal.py.

Запуск:
    python helper_scripts/debug_bbox.py "data/РЦ-АП25 ТС МОС топливных систем самолетов.pdf" 6

Второй аргумент — номер страницы (1-based).
"""

import sys
from pathlib import Path

import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# PyMuPDF
# ---------------------------------------------------------------------------

def get_text_blocks_with_bbox(pdf_path, page_no):
    """Возвращает (blocks, page_size). blocks — список {'bbox', 'text'}."""
    doc = pymupdf.open(pdf_path)
    page = doc[page_no - 1]
    page_size = (page.rect.width, page.rect.height)

    blocks = []
    for b in page.get_text("blocks"):
        x0, y0, x1, y1, text, block_no, block_type = b
        if block_type != 0:
            continue
        text = text.strip()
        if not text:
            continue
        blocks.append({
            "bbox": (float(x0), float(y0), float(x1), float(y1)),
            "text": text,
        })
    doc.close()
    return blocks, page_size


# ---------------------------------------------------------------------------
# Docling
# ---------------------------------------------------------------------------

def get_tables_with_docling(pdf_path, page_no):
    """Возвращает список таблиц страницы с BBox в TOPLEFT (points)."""
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.do_ocr = False
    pipeline_options.table_structure_options.do_cell_matching = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    result = converter.convert(str(pdf_path))

    # Высота страницы для перевода BOTTOMLEFT → TOPLEFT
    doc = pymupdf.open(pdf_path)
    page_height = doc[page_no - 1].rect.height
    doc.close()

    tables = []
    for table in result.document.tables:
        # Фильтр по label: только реальные таблицы (не оглавления)
        label_str = str(getattr(table, "label", "")).lower()
        if "table" not in label_str or "document_index" in label_str:
            continue

        if not table.prov:
            continue
        prov = table.prov[0]
        if prov.page_no != page_no:
            continue


        # BBox с нормализацией
        bbox = None
        if prov.bbox is not None:
            bb = prov.bbox

            # Шаг 1: привести к TOPLEFT
            if bb.coord_origin.name == "BOTTOMLEFT":
                y0 = page_height - float(bb.b)  # верх = H - bottom
                y1 = page_height - float(bb.t)  # низ  = H - top
            else:
                y0 = float(bb.t)
                y1 = float(bb.b)

            # Шаг 2: гарантировать y0 < y1 (иногда Docling отдаёт перепутанные t/b)
            if y0 > y1:
                y0, y1 = y1, y0

            bbox = (float(bb.l), y0, float(bb.r), y1)

        # Markdown с doc (чтобы не было deprecated warning)
        try:
            markdown = table.export_to_markdown(doc=result.document)
        except TypeError:
            markdown = table.export_to_markdown()

        # Caption: пробуем через captions-рефы
        caption = ""
        try:
            cap_list = getattr(table, "captions", []) or []
            parts = []
            for ref in cap_list:
                cref = getattr(ref, "cref", None)
                if not cref:
                    continue
                # cref вида "#/texts/38"
                try:
                    _, _, idx = cref.split("/")
                    item = result.document.texts[int(idx)]
                    parts.append(str(item.text))
                except Exception:
                    pass
            caption = " ".join(parts).strip()
        except Exception:
            pass

        tables.append({
            "bbox": bbox,
            "markdown": markdown,
            "caption": caption,
            "label": label_str,
        })
    return tables


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print('Использование: python debug_bbox.py <pdf> <page_no>')
        print('Пример:  python debug_bbox.py "data/РЦ-АП25 ТС МОС топливных систем самолетов.pdf" 6')
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    page_no = int(sys.argv[2])

    if not pdf_path.is_absolute():
        pdf_path = PROJECT_ROOT / pdf_path

    if not pdf_path.exists():
        print(f"Файл не найден: {pdf_path}")
        sys.exit(1)

    print(f"PDF:        {pdf_path.name}")
    print(f"Страница:   {page_no}\n")

    # --- PyMuPDF ---
    print("=" * 72)
    print("PyMuPDF: текстовые блоки")
    print("=" * 72)
    blocks, page_size = get_text_blocks_with_bbox(pdf_path, page_no)
    print(f"Размер страницы (points): {page_size[0]:.1f} x {page_size[1]:.1f}")
    print(f"Всего блоков: {len(blocks)}\n")

    for i, blk in enumerate(blocks[:10]):
        x0, y0, x1, y1 = blk["bbox"]
        preview = blk["text"][:70].replace("\n", " ")
        print(f"  [{i}] bbox=({x0:7.1f}, {y0:7.1f}, {x1:7.1f}, {y1:7.1f})")
        print(f"      {preview!r}")

    if len(blocks) > 10:
        print(f"  ... и ещё {len(blocks) - 10} блоков")

    # --- Docling ---
    print("\n" + "=" * 72)
    print("Docling: таблицы")
    print("=" * 72)
    try:
        tables = get_tables_with_docling(pdf_path, page_no)
    except ImportError as e:
        print(f"⚠ Docling не установлен: {e}")
        return
    except Exception as e:
        print(f"⚠ Docling упал: {e}")
        return

    print(f"Всего таблиц на странице: {len(tables)}\n")
    for i, tbl in enumerate(tables):
        print(f"  Таблица [{i}] (label={tbl.get('label','?')}):")
        print(f"    bbox (TOPLEFT): {tbl['bbox']}")
        print(f"    caption:        {tbl['caption']!r}")
        md = (tbl["markdown"] or "")[:200].replace("\n", "\\n")
        print(f"    markdown:       {md!r}")
        print()

    # --- Диагностика ---
    print("=" * 72)
    print("Диагностика")
    print("=" * 72)

    if not blocks:
        print("⚠ PyMuPDF не нашёл текстовых блоков на этой странице.")
    if not tables:
        print("⚠ Docling не нашёл таблиц на этой странице.")
        print("   Попробуйте другую страницу (там, где явно есть таблица).")

    if blocks and tables:
        # Проверяем, что BBox таблицы реально пересекается с блоками PyMuPDF
        tbl_y_min = min(t["bbox"][1] for t in tables if t["bbox"])
        tbl_y_max = max(t["bbox"][3] for t in tables if t["bbox"])
        blk_y_min = min(b["bbox"][1] for b in blocks)
        blk_y_max = max(b["bbox"][3] for b in blocks)

        print(f"Диапазон Y блоков PyMuPDF:  {blk_y_min:.1f} … {blk_y_max:.1f}")
        print(f"Диапазон Y таблиц Docling:  {tbl_y_min:.1f} … {tbl_y_max:.1f}")

        # Таблица должна попадать внутрь диапазона страницы
        if blk_y_min <= tbl_y_min and tbl_y_max <= blk_y_max * 1.05:
            print("✅ BBox таблицы Docling согласуется с координатами PyMuPDF.")
            print("   overlaps() можно использовать как есть.")
        else:
            print("⚠ BBox таблицы Docling выходит за диапазон блоков PyMuPDF.")
            print("   Возможно, нужна дополнительная калибровка.")


if __name__ == "__main__":
    main()