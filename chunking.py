import json
import re
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

INPUT_PATH = Path("ocr_pages_v2.json")
OUTPUT_PATH = Path("chunks.json")

# --- Параметры чанкования ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# --- Разделители (в порядке приоритета) ---
# Сначала пытаемся резать по границам пунктов, потом по абзацам, строкам, пробелам.
SEPARATORS = [
    r"\n(?=\d+\.\d+\.)",   # перед "4.1.", "3.2."
    r"\n(?=\d+\.)",        # перед "1.", "2."
    r"\n(?=РАЗДЕЛ)",       # перед "РАЗДЕЛ А", "РАЗДЕЛ Б"
    r"\n(?=ПРИЛОЖЕНИЕ)",   # перед "ПРИЛОЖЕНИЕ А"
    r"\n\n",               # обычные абзацы
    r"\n",                 # строки
    " ",                   # слова
    "",                    # символы
]


def clean_text(text: str) -> str:
    """Минимальная чистка OCR-текста."""
    # Склеиваем переносы слов
    text = re.sub(r"-\n(\w)", r"\1", text)
    # Множественные пробелы
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Убираем пробелы в начале строк
    text = re.sub(r"\n +", "\n", text)
    return text.strip()


def build_offset_map(pages):
    """
    Строит карту: для каждой content-страницы — (start_offset, end_offset, page_num).
    Возвращает также склеенный текст.
    """
    offset_map = []
    parts = []
    cursor = 0
    for p in pages:
        if p["type"] != "content":
            continue
        cleaned = clean_text(p["text"])
        if not cleaned:
            continue
        start = cursor
        parts.append(cleaned)
        cursor += len(cleaned) + 1  # +1 за '\n' между страницами
        end = cursor
        offset_map.append((start, end, p["page"]))
    full_text = "\n".join(parts)
    return full_text, offset_map


def pages_for_range(start, end, offset_map):
    """Какие страницы покрывает диапазон [start, end)."""
    covered = []
    for (s, e, page_num) in offset_map:
        if s < end and e > start:
            covered.append(page_num)
    return covered

def merge_small_chunks(chunks, min_size=150):
    """
    Сливает чанки короче min_size с соседними.
    Логика: маленький чанк присоединяется к следующему (или к предыдущему, если следующего нет).
    """
    if not chunks:
        return chunks

    merged = []
    i = 0
    while i < len(chunks):
        current = chunks[i].copy()

        # Если текущий короткий и есть следующий — сливаем со следующим
        if current["n_chars"] < min_size and i + 1 < len(chunks):
            nxt = chunks[i + 1]
            current["text"] = current["text"].rstrip() + "\n\n" + nxt["text"].lstrip()
            current["n_chars"] = len(current["text"])
            # Объединяем pages (сохраняя порядок и уникальность)
            all_pages = sorted(set(current["pages"] + nxt["pages"]))
            current["pages"] = all_pages
            current["page_span"] = (
                str(all_pages[0]) if len(all_pages) == 1
                else f"{all_pages[0]}–{all_pages[-1]}"
            )
            # Тип — content (если хоть один был content)
            if current["type"] == "title" or nxt["type"] == "title":
                current["type"] = "title"
            else:
                current["type"] = "content"
            i += 2  # пропускаем оба
        else:
            i += 1

        merged.append(current)

    # Пересчитываем chunk_id
    for new_id, c in enumerate(merged):
        c["chunk_id"] = new_id

    return merged


if __name__ == "__main__":
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        pages = json.load(f)

    # --- 1. Отдельно обрабатываем титульник ---
    chunks = []
    chunk_id = 0
    for p in pages:
        if p["type"] == "title":
            chunks.append({
                "chunk_id": chunk_id,
                "text": clean_text(p["text"]),
                "pages": [p["page"]],
                "page_span": str(p["page"]),
                "type": "title",
                "n_chars": len(clean_text(p["text"])),
            })
            chunk_id += 1

    # --- 2. Склеиваем все content-страницы ---
    full_text, offset_map = build_offset_map(pages)
    print(f"Склеено content-страниц: {len(offset_map)}")
    print(f"Длина склеенного текста: {len(full_text)} симв.")

    # --- 3. Режем на чанки ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        is_separator_regex=True,
        length_function=len,
    )

    # Используем create_documents, чтобы получить смещения
    split_docs = splitter.create_documents([full_text])

    # Пересчитываем оффсеты для каждого чанка
    # (create_documents не даёт offsets, найдём их поиском)
    search_start = 0
    for doc in split_docs:
        chunk_text = doc.page_content.strip()
        idx = full_text.find(chunk_text, search_start)
        if idx == -1:
            # fallback: ищем без strip
            idx = full_text.find(doc.page_content, search_start)
        chunk_start = idx
        chunk_end = idx + len(doc.page_content)
        search_start = max(chunk_start, chunk_end - CHUNK_OVERLAP)  # чтобы не искать с нуля

        covered_pages = pages_for_range(chunk_start, chunk_end, offset_map)
        pages_str = (
            str(covered_pages[0]) if len(covered_pages) == 1
            else f"{covered_pages[0]}–{covered_pages[-1]}"
        )

        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "pages": covered_pages,
            "page_span": pages_str,
            "type": "content",
            "n_chars": len(chunk_text),
        })
        chunk_id += 1

    chunks = merge_small_chunks(chunks, min_size=150)

    # --- 4. Сохраняем ---
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    # --- 5. Статистика ---
    print(f"\nВсего чанков: {len(chunks)}")
    print(f"   из них title:   {sum(1 for c in chunks if c['type'] == 'title')}")
    print(f"   из них content: {sum(1 for c in chunks if c['type'] == 'content')}")

    sizes = [c["n_chars"] for c in chunks if c["type"] == "content"]
    if sizes:
        print(f"\nРазмеры content-чанков (символов):")
        print(f"   min:    {min(sizes)}")
        print(f"   max:    {max(sizes)}")
        print(f"   средн.: {sum(sizes) / len(sizes):.0f}")

    multi_page = [c for c in chunks if len(c["pages"]) > 1]
    print(f"\nЧанков, пересекающих границы страниц: {len(multi_page)}")

    print(f"\nСохранено в {OUTPUT_PATH}")

    # --- 6. Показываем первые 3 чанка для проверки ---
    print("\n" + "=" * 70)
    print(" Первые 3 чанка (для визуальной проверки)")
    print("=" * 70)
    for c in chunks[:3]:
        print(f"\n--- chunk_id={c['chunk_id']}, pages={c['page_span']}, type={c['type']} ---")
        print(c["text"][:300])
        if len(c["text"]) > 300:
            print("...")