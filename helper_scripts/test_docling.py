import sys
import time
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions,
)

TESSDATA_DIR = r"C:\Users\l.ilyintseva\Downloads\tesseract-ocr-w64-setup-5.5.3.20260724\tessdata"

ocr_options = TesseractCliOcrOptions(
    lang=["rus", "eng"],
    tesseract_cmd=r"C:\Users\l.ilyintseva\Downloads\tesseract-ocr-w64-setup-5.5.3.20260724\tesseract.exe",
    # Указываем папку с traineddata
    path=TESSDATA_DIR,
)

def main():
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = BASE_DIR.parent
    DATA_DIR = PROJECT_ROOT / "data"
    REPORTS_DIR = PROJECT_ROOT / "reports"

    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (DATA_DIR / "РЦ-АП25 ТС МОС топливных систем самолетов.pdf")

    if not pdf_path.exists():
        print(f" Файл не найден: {pdf_path}")
        return

    print(f" Файл: {pdf_path.name}")
    print(f" Запускаю Docling с Tesseract CLI...")

    # --- Tesseract CLI (без tesserocr) ---
    ocr_options = TesseractCliOcrOptions(
        lang=["rus", "eng"],  # ISO 639-2 коды
        tesseract_cmd=r"C:\Users\l.ilyintseva\Downloads\tesseract-ocr-w64-setup-5.5.3.20260724\tesseract.exe",
    )

    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        ocr_options=ocr_options,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    t0 = time.time()
    result = converter.convert(pdf_path)
    doc = result.document

    elapsed = time.time() - t0
    print(f" Готово за {elapsed:.1f} сек\n")

    # --- Статистика ---
    print("=" * 70)
    print(" СТАТИСТИКА")
    print("=" * 70)
    print(f"   Текстовых блоков: {len(doc.texts)}")
    print(f"   Таблиц:           {len(doc.tables)}")
    print(f"   Картинок:         {len(doc.pictures)}")

    # --- Показать все таблицы ---
    if doc.tables:
        print("\n" + "=" * 70)
        print(" ТАБЛИЦЫ (markdown)")
        print("=" * 70)
        for i, table in enumerate(doc.tables):
            print(f"\n--- Таблица {i + 1} ---")
            try:
                df = table.export_to_dataframe(doc=doc)
                print(f"Форма: {df.shape}")
                print(df.to_markdown(index=False))
            except Exception as e:
                print(f"️ Не удалось вывести как markdown: {e}")

    # --- Первые 3 текстовых блока ---
    print("\n" + "=" * 70)
    print(" ПЕРВЫЕ 3 ТЕКСТОВЫХ БЛОКА")
    print("=" * 70)
    for i, item in enumerate(doc.texts[:3]):
        print(f"\n--- Блок {i + 1} ---")
        print(item.text[:500])

    # --- Сохранить полный markdown ---
    REPORTS_DIR.mkdir(exist_ok=True)
    output_path = REPORTS_DIR / "docling_output.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(doc.export_to_markdown())
    print(f"\n Полный Markdown сохранён в {output_path}")


if __name__ == "__main__":
    main()