"""
Чанкование новых документов.

Режимы:
    python chunking.py                # обработать ВСЕ документы (перезапись chunks.json)
    python chunking.py --add          # обработать только те, которых нет в chunks.json

По умолчанию (без аргументов) — ПОЛНАЯ пересборка chunks.json из pages.json.
С флагом --add — добавляются только НОВЫЕ документы, старые не трогаются.
"""

import argparse
import json
import re
from pathlib import Path
from collections import Counter

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Конфигурация
PAGES_PATH = Path("pages.json")
CHUNKS_PATH = Path("chunks.json")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MIN_CHUNK_SIZE = 150

SEPARATORS = [
    r"\n(?=\d+\.\d+\.)",
    r"\n(?=\d+\.)",
    r"\n(?=РАЗДЕЛ)",
    r"\n(?=ПРИЛОЖЕНИЕ)",
    r"\n\n",
    r"\n",
    " ",
    "",
]


# Утилиты
def clean_text(text: str) -> str:
    """Минимальная чистка OCR-текста."""
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n +", "\n", text)
    return text.strip()


def build_offset_map(pages):
    """Карта оффсетов для content-страниц: (start, end, page_num)."""
    offset_map = []
    parts = []
    cursor = 0
    for p in pages:
        if p.get("type") != "content":
            continue
        cleaned = clean_text(p["text"])
        if not cleaned:
            continue
        start = cursor
        parts.append(cleaned)
        cursor += len(cleaned) + 1
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


def merge_small_chunks(chunks, min_size=MIN_CHUNK_SIZE):
    """Сливает чанки короче min_size с соседними."""
    if not chunks:
        return chunks

    merged = []
    i = 0
    while i < len(chunks):
        current = chunks[i].copy()
        if current["n_chars"] < min_size and i + 1 < len(chunks):
            nxt = chunks[i + 1]
            current["text"] = current["text"].rstrip() + "\n\n" + nxt["text"].lstrip()
            current["n_chars"] = len(current["text"])
            all_pages = sorted(set(current["pages"] + nxt["pages"]))
            current["pages"] = all_pages
            current["page_span"] = (
                str(all_pages[0]) if len(all_pages) == 1
                else f"{all_pages[0]}–{all_pages[-1]}"
            )
            if current["type"] == "title" or nxt["type"] == "title":
                current["type"] = "title"
            else:
                current["type"] = "content"
            i += 2
        else:
            i += 1
        merged.append(current)

    return merged

# Ядро чанкования одного документа
def chunk_document(pages, source, start_chunk_id):
    """Чанкует страницы одного документа."""
    chunks = []
    chunk_id = start_chunk_id

    # 1. Титульники — отдельными чанками
    for p in pages:
        if p.get("type") == "title":
            cleaned = clean_text(p["text"])
            if not cleaned:
                continue
            chunks.append({
                "chunk_id": chunk_id,
                "text": cleaned,
                "pages": [p["page"]],
                "page_span": str(p["page"]),
                "type": "title",
                "n_chars": len(cleaned),
                "source": source,
            })
            chunk_id += 1

    # 2. Content-страницы
    content_pages = [p for p in pages if p.get("type") == "content"]
    if content_pages:
        full_text, offset_map = build_offset_map(content_pages)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=SEPARATORS,
            is_separator_regex=True,
            length_function=len,
        )
        split_docs = splitter.create_documents([full_text])

        search_start = 0
        for doc in split_docs:
            chunk_text = doc.page_content.strip()
            idx = full_text.find(chunk_text, search_start)
            if idx == -1:
                idx = full_text.find(doc.page_content, search_start)
            chunk_start = idx
            chunk_end = idx + len(doc.page_content)
            search_start = max(chunk_start, chunk_end - CHUNK_OVERLAP)

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
                "source": source,
            })
            chunk_id += 1

    # 3. Слияние маленьких
    chunks = merge_small_chunks(chunks, min_size=MIN_CHUNK_SIZE)

    # Пересчитываем chunk_id
    for i, c in enumerate(chunks):
        c["chunk_id"] = start_chunk_id + i

    return chunks



def get_existing_sources(chunks):
    """Множество source из существующих чанков."""
    return {c.get("source") for c in chunks if "source" in c}



def rebuild_all(pages_data):
    """Пересобирает chunks.json из ВСЕХ документов pages.json."""
    all_chunks = []
    start_id = 0

    for doc_name, doc in pages_data.items():
        pages = doc["pages"]
        if any("type" not in p for p in pages):
            print(f"Документ '{doc_name}' не классифицирован. Пропускаю.")
            continue

        doc_chunks = chunk_document(pages, source=doc_name, start_chunk_id=start_id)
        print(f"   Чанков: {len(doc_chunks)}")
        all_chunks.extend(doc_chunks)
        start_id = len(all_chunks)

    return all_chunks

def add_new(pages_data, existing_chunks):
    """Добавляет чанки для документов, которых нет в existing_chunks."""
    existing_sources = get_existing_sources(existing_chunks)

    # Определяем, какие документы новые
    new_docs = []
    for doc_name, doc in pages_data.items():
        if doc_name in existing_sources:
            continue
        if any("type" not in p for p in doc["pages"]):
            print(f" '{doc_name}' не классифицирован. Пропускаю.")
            continue
        new_docs.append(doc_name)

    if not new_docs:
        print("\nВсе документы из pages.json уже прочанкованы. Нечего делать.")
        return None

    print(f"\nНовых документов: {len(new_docs)}")
    for name in new_docs:
        print(f"   • {name}")


    start_id = max((c["chunk_id"] for c in existing_chunks), default=-1) + 1
    print(f"\n   Начинаю с chunk_id = {start_id}")

    new_chunks = []
    for doc_name in new_docs:
        pages = pages_data[doc_name]["pages"]
        print(f"\n Чанкую '{doc_name}'...")
        doc_chunks = chunk_document(pages, source=doc_name, start_chunk_id=start_id)
        print(f"   Чанков: {len(doc_chunks)}")
        new_chunks.extend(doc_chunks)
        start_id += len(doc_chunks)

    return new_chunks



def main():
    parser = argparse.ArgumentParser(description="Чанкование документов")
    parser.add_argument(
        "--add", action="store_true",
        help="Обработать только НОВЫЕ документы (не пересобирать всё)"
    )
    args = parser.parse_args()

    if not PAGES_PATH.exists():
        print(f"{PAGES_PATH} не найден.")
        return

    print(f" Читаю {PAGES_PATH}...")
    with open(PAGES_PATH, "r", encoding="utf-8") as f:
        pages_data = json.load(f)

    total_pages = sum(len(d["pages"]) for d in pages_data.values())
    print(f"   Документов: {len(pages_data)}")
    print(f"   Всего страниц: {total_pages}")

    if args.add:
        #  Режим --add: только новые
        if not CHUNKS_PATH.exists():
            print(f"\n {CHUNKS_PATH} не найден — запустите без --add для первичной сборки.")
            return

        print(f"\n Читаю {CHUNKS_PATH}...")
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            existing_chunks = json.load(f)
        print(f"   Существующих чанков: {len(existing_chunks)}")

        new_chunks = add_new(pages_data, existing_chunks)
        if new_chunks is None:
            return

        all_chunks = existing_chunks + new_chunks

        # Бэкап
        backup_path = CHUNKS_PATH.with_name("chunks.backup.json")
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(existing_chunks, f, ensure_ascii=False, indent=2)
        print(f"\n Бэкап: {backup_path}")

    else:
        # Полная пересборка
        print(f"\n Полная пересборка chunks.json...")
        all_chunks = rebuild_all(pages_data)

        # Бэкап старого chunks.json
        if CHUNKS_PATH.exists():
            backup_path = CHUNKS_PATH.with_name("chunks.backup.json")
            with open(CHUNKS_PATH, "r", encoding="utf-8") as f_src:
                old = json.load(f_src)
            with open(backup_path, "w", encoding="utf-8") as f_dst:
                json.dump(old, f_dst, ensure_ascii=False, indent=2)
            print(f"\nБэкап: {backup_path}")

    # --- Сохраняем ---
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"\n Сохранено: {CHUNKS_PATH}")

    # --- Сводка ---
    print(f"\n Всего чанков: {len(all_chunks)}")
    types = Counter(c["type"] for c in all_chunks)
    print(f"   По типам: {dict(types)}")

    sizes = [c["n_chars"] for c in all_chunks if c["type"] == "content"]
    if sizes:
        print(f"   Размеры content: min={min(sizes)}, max={max(sizes)}, "
              f"avg={sum(sizes) / len(sizes):.0f}")

    print(f"\nПо источникам:")
    sources = Counter(c["source"] for c in all_chunks)
    for src, cnt in sorted(sources.items()):
        print(f"   • {src}: {cnt} чанков")


if __name__ == "__main__":
    main()