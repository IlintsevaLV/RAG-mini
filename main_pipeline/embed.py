"""
Эмбеддинги + FAISS.

Режимы:
    python embed.py --add       # добавить эмбеддинги только для новых чанков (в конце списка)
    python embed.py --rebuild   # пересобрать всё с нуля

ВАЖНО: если chunks.json пересобран полностью (например, после изменения chunking.py),
используйте ТОЛЬКО --rebuild. Режим --add добавляет чанки, которые оказались в конце
chunks.json, и не подходит, если новые чанки вклинились в середину.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Отключаем проверку обновлений HF (модель уже в кеше)
os.environ["HF_HUB_OFFLINE"] = "1"

# Конфигурация
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
CHUNKS_PATH = BASE_DIR / "chunks.json"
EMBEDDINGS_PATH = BASE_DIR / "embeddings.npy"
INDEX_PATH = BASE_DIR / "faiss.index"
MODEL_PATH = PROJECT_ROOT / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"

EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"
BATCH_SIZE = 16
MAX_SEQ_LENGTH = 512   # лимит E5-large; всё, что длиннее, обрезается


# ---------------------------------------------------------------------------
# Диагностика
# ---------------------------------------------------------------------------

def check_token_lengths(model, chunks):
    """Проверяет, не превышают ли чанки лимит токенов E5.

    Возвращает список превышающих чанков (для отладки).
    """
    tokenizer = model.tokenizer
    max_len = model.max_seq_length

    over_limit = []
    for c in chunks:
        text = f"passage: {c['text']}"
        n_tokens = len(tokenizer.encode(text, add_special_tokens=True))
        if n_tokens > max_len:
            over_limit.append({
                "chunk_id": c["chunk_id"],
                "type": c.get("type", "?"),
                "tokens": n_tokens,
                "chars": c["n_chars"],
            })

    if over_limit:
        print(f"\n  Чанки, превышающие {max_len} токенов: {len(over_limit)}")
        # Показываем первые 10
        for item in over_limit[:10]:
            print(f"     chunk_id={item['chunk_id']}, type={item['type']}, "
                  f"tokens={item['tokens']}, chars={item['chars']}")
        if len(over_limit) > 10:
            print(f"     ... и ещё {len(over_limit) - 10}")

        # Разбивка по типу
        from collections import Counter
        by_type = Counter(item["type"] for item in over_limit)
        print(f"     По типам: {dict(by_type)}")
    else:
        print(f"\n  Все чанки в пределах {max_len} токенов.")

    return over_limit


# ---------------------------------------------------------------------------
# Полная пересборка
# ---------------------------------------------------------------------------

def rebuild_all(model):
    """Пересобирает embeddings.npy и faiss.index с нуля по всем чанкам."""
    print(f"\n Читаю {CHUNKS_PATH}...")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"   Чанков: {len(chunks)}")

    # Диагностика длины токенов
    check_token_lengths(model, chunks)

    # Тексты с префиксом "passage: "
    texts = [f"passage: {c['text']}" for c in chunks]

    print(f"\n Считаю эмбеддинги для {len(texts)} чанков...")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    print(f"   Готово за {time.time() - t0:.1f} сек")
    print(f"   Форма: {embeddings.shape}")

    # Бэкапы
    if EMBEDDINGS_PATH.exists():
        backup = EMBEDDINGS_PATH.with_name("embeddings.backup.npy")
        np.save(backup, np.load(EMBEDDINGS_PATH))
        print(f" Бэкап: {backup}")
    if INDEX_PATH.exists():
        backup = INDEX_PATH.with_name("faiss.backup.index")
        faiss.write_index(faiss.read_index(str(INDEX_PATH)), str(backup))
        print(f" Бэкап: {backup}")

    # Сохраняем
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f" Сохранено: {EMBEDDINGS_PATH}")

    # FAISS
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, str(INDEX_PATH))
    print(f" Сохранено: {INDEX_PATH} ({index.ntotal} векторов)")

    return len(chunks), embeddings.shape


# ---------------------------------------------------------------------------
# Инкрементальное добавление
# ---------------------------------------------------------------------------

def add_new(model):
    """Добавляет эмбеддинги только для НОВЫХ чанков В КОНЦЕ списка.

    ВНИМАНИЕ: подходит только если новые чанки лежат строго после существующих
    (например, при добавлении нового документа через chunks.json --add).
    Для пересобранного chunks.json используйте --rebuild.
    """
    print(f"\n Читаю {CHUNKS_PATH}...")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"   Всего чанков: {len(chunks)}")

    if not EMBEDDINGS_PATH.exists():
        print(f"\n {EMBEDDINGS_PATH} не найден.")
        print(f"   Запустите без --add для первичной сборки.")
        return None

    embeddings = np.load(EMBEDDINGS_PATH)
    print(f"   Существующих эмбеддингов: {embeddings.shape[0]}")
    print(f"   Размерность: {embeddings.shape[1]}")

    n_existing = embeddings.shape[0]
    n_total = len(chunks)
    n_new = n_total - n_existing

    if n_new == 0:
        print(f"\n Все чанки уже добавлены в эмбеддинг.")
        return None

    if n_new < 0:
        print(f"\n Внимание: эмбеддингов больше, чем чанков ({n_existing} > {n_total}).")
        print(f"   Возможно, chunks.json изменился. Запустите с --rebuild.")
        return None

    print(f"\n Новых чанков: {n_new}")
    print(f"   (с {n_existing}-го по {n_total - 1}-й)")

    new_chunks = chunks[n_existing:]
    new_texts = [f"passage: {c['text']}" for c in new_chunks]

    new_sources = set(c.get("source") for c in new_chunks)
    print(f"   Источники новых чанков: {new_sources}")

    print(f"\n Считаю эмбеддинги для {len(new_texts)} новых чанков...")
    t0 = time.time()
    new_embeddings = model.encode(
        new_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    print(f"   Готово за {time.time() - t0:.1f} сек")

    # Бэкапы
    emb_backup = EMBEDDINGS_PATH.with_name("embeddings.backup.npy")
    np.save(emb_backup, embeddings)
    print(f" Бэкап: {emb_backup}")

    idx_backup = INDEX_PATH.with_name("faiss.backup.index")
    index = faiss.read_index(str(INDEX_PATH))
    faiss.write_index(index, str(idx_backup))
    print(f" Бэкап: {idx_backup}")

    # Дописываем к embeddings.npy
    all_embeddings = np.vstack([embeddings, new_embeddings])
    np.save(EMBEDDINGS_PATH, all_embeddings)
    print(f" Сохранено: {EMBEDDINGS_PATH}")
    print(f"   Было: {n_existing}, стало: {all_embeddings.shape[0]}")

    # Дописываем в FAISS
    print(f"   Было в FAISS: {index.ntotal}")
    index.add(new_embeddings.astype(np.float32))
    faiss.write_index(index, str(INDEX_PATH))
    print(f" Сохранено: {INDEX_PATH}")
    print(f"   Стало в FAISS: {index.ntotal}")

    return n_new, all_embeddings.shape


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Эмбеддинги + FAISS")
    parser.add_argument(
        "--add", action="store_true",
        help="Добавить эмбеддинги только для новых чанков (в конце списка)"
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Пересобрать всё с нуля"
    )
    args = parser.parse_args()

    if not args.add and not args.rebuild:
        print("  Укажите режим: --add или --rebuild")
        print("   python embed.py --add       (только новые)")
        print("   python embed.py --rebuild   (всё заново)")
        return

    # Загружаем модель
    print(f" Загружаю модель {EMBED_MODEL_NAME}...")
    t0 = time.time()
    model = SentenceTransformer(EMBED_MODEL_NAME)
    model.max_seq_length = MAX_SEQ_LENGTH
    print(f"   Готово за {time.time() - t0:.1f} сек")
    print(f"   max_seq_length = {model.max_seq_length}")

    if args.rebuild:
        result = rebuild_all(model)
    else:
        result = add_new(model)

    if result is None:
        return

    # Финальная проверка
    print(f"\n Проверка:")
    final_embeddings = np.load(EMBEDDINGS_PATH)
    final_index = faiss.read_index(str(INDEX_PATH))
    print(f"   embeddings.npy: {final_embeddings.shape}")
    print(f"   faiss.index:    {final_index.ntotal} векторов")

    if final_embeddings.shape[0] == final_index.ntotal:
        print(f"   Согласовано")
    else:
        print(f"   НЕСОГЛАСОВАНО! Проверьте файлы.")


if __name__ == "__main__":
    main()