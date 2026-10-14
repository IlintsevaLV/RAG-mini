"""
Тест: Docling с TableFormerMode.ACCURATE — сравнение таблиц.
"""
import sys
import time
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DIR / "РЦ-АП25 ТС МОС топливных систем самолетов.pdf"

print(f"📄 {pdf_path.name}")
print("🚀 Docling + TableFormerMode.ACCURATE...")

pipeline_options = PdfPipelineOptions(
    do_table_structure=True,
    table_structure_options=TableStructureOptions(
        mode=TableFormerMode.ACCURATE,
        do_cell_matching=True,
    ),
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

print(f"✅ Готово за {elapsed:.1f} сек")
print(f"📊 Таблиц: {len(doc.tables)}")

# Первые 5 таблиц с page_no
for i, table in enumerate(doc.tables[:5]):
    prov = getattr(table, "prov", None)
    page_no = None
    if prov and isinstance(prov, list) and len(prov) > 0:
        page_no = getattr(prov[0], "page_no", None)
    print(f"\n--- Таблица {i+1} (стр. {page_no}) ---")
    try:
        df = table.export_to_dataframe(doc=doc)
        print(f"Форма: {df.shape}")
        print(df.head(5).to_string())
    except Exception as e:
        print(f"⚠️ Ошибка: {e}")

# Сохранить в reports
REPORTS_DIR.mkdir(exist_ok=True)
output_path = REPORTS_DIR / "docling_accurate.md"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(doc.export_to_markdown())
print(f"\n💾 Сохранено: {output_path}")