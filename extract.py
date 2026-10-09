import pypdf
from pathlib import Path

PDF_PATH = Path(r"C:/Users/l.ilyintseva/Desktop/RAG-mini/data/Руководство 23-29.605 по МОС образцов ВС.pdf")

def extract_pages(pdf_path):
    reader = pypdf.PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({
            "page": i + 1,          # нумерация с 1
            "text": text,
            "n_chars": len(text),
        })
    return pages

if __name__ == "__main__":
    if not PDF_PATH.exists():
        print(f"Файл не найден: {PDF_PATH}")
        raise SystemExit(1)

    pages = extract_pages(PDF_PATH)
    print(f" Всего страниц: {len(pages)}")
    print(f" Всего символов: {sum(p['n_chars'] for p in pages)}")
    print()

    # Покажем статистику по страницам
    for p in pages:
        bar = "█" * min(60, p["n_chars"] // 100)
        print(f"стр. {p['page']:>2}: {p['n_chars']:>5} симв. {bar}")

    print("\n" + "=" * 70)
    print(" Первые 800 символов со страницы 1:")
    print("=" * 70)
    print(pages[0]["text"][:800])

    print("\n" + "=" * 70)
    print(" Первые 800 символов со страницы 5:")
    print("=" * 70)
    print(pages[4]["text"][:800])