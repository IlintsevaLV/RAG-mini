"""
Прогон VLM на скане через Docling.
Использует лёгкую модель Granite-Docling-258M.
"""

import sys
import time
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
from docling.pipeline.vlm_pipeline import VlmPipeline


def main():
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "data/Руководство 23-29.605 по МОС образцов ВС.pdf"
    )

    if not pdf_path.exists():
        print(f" Файл не найден: {pdf_path}")
        return

    print(f" Файл: {pdf_path.name}")
    print(" Запускаю VLM (Granite-Docling-258M, CPU)...")

    # --- Настройки VLM ---
    vlm_options = VlmConvertOptions.from_preset("granite_docling")
    # Можно увеличить разрешение для лучшего качества (медленнее):
    # vlm_options.scale = 2.0
    # vlm_options.max_size = 1024

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=VlmPipelineOptions(vlm_options=vlm_options),
            )
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

    # --- Первые 3 текстовых блока ---
    print("\n" + "=" * 70)
    print(" ПЕРВЫЕ 3 ТЕКСТОВЫХ БЛОКА")
    print("=" * 70)
    for i, item in enumerate(doc.texts[:3]):
        print(f"\n--- Блок {i + 1} ---")
        print(item.text[:500])

    # --- Сохранить полный markdown ---
    output_path = Path("vlm_output.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(doc.export_to_markdown())
    print(f"\n Полный Markdown сохранён в {output_path}")


if __name__ == "__main__":
    main()