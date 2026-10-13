"""
Инкрементальное добавление эмбеддингов.

Режимы:
    python add_embeddings.py --add    # добавить эмбеддинги для новых чанков
    python add_embeddings.py --rebuild  # пересобрать всё с нуля

"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Отключаем проверку обновлений HF
os.environ["HF_HUB_OFFLINE"] = "1"

# Конфигурация
CHUNKS_PATH = Path("chunks.json")
EMBEDDINGS_PATH = Path("embeddings.npy")
INDEX_PATH = Path("faiss.index")

EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"
BATCH_SIZE = 16



# Полная пересборка
def rebuild_all(model):
    """Пересобирает embeddings.npy и faiss.index с нуля по всем чанкам."""
    print(f"\n Читаю {CHUNKS_PATH}...")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"   Чанков: {len(chunks)}")

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


# Инкрементальное добавление
def add_new(model):
    """Добавляет эмбеддинги только для новых чанков."""
    #1. Читаем chunks.json
    print(f"\n Читаю {CHUNKS_PATH}...")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"   Всего чанков: {len(chunks)}")

    #2. Читаем существующие эмбеддинги
    if not EMBEDDINGS_PATH.exists():
        print(f"\n {EMBEDDINGS_PATH} не найден.")
        print(f"   Запустите без --add для первичной сборки.")
        return None

    embeddings = np.load(EMBEDDINGS_PATH)
    print(f"   Существующих эмбеддингов: {embeddings.shape[0]}")
    print(f"   Размерность: {embeddings.shape[1]}")

    # 3. Определяем, сколько нужно добавить
    n_existing = embeddings.shape[0]
    n_total = len(chunks)
    n_new = n_total - n_existing

    if n_new <= 0:
        print(f"\n Все чанки уже добавлены в эмбеддинг.")
        return None

    if n_new < 0:
        print(f"\n Внимание: эмбеддингов больше, чем чанков ({n_existing} > {n_total}).")
        print(f"   Возможно, chunks.json изменился. Запустите без --add для пересборки.")
        return None

    print(f"\n Новых чанков: {n_new}")
    print(f"   (с {n_existing}-го по {n_total - 1}-й)")

    #4. Считаем эмбеддинги только для новых
    new_chunks = chunks[n_existing:]
    new_texts = [f"passage: {c['text']}" for c in new_chunks]

    # Проверим, что это действительно новые (по source)
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

    #5. Бэкапы
    emb_backup = EMBEDDINGS_PATH.with_name("embeddings.backup.npy")
    np.save(emb_backup, embeddings)
    print(f" Бэкап: {emb_backup}")

    idx_backup = INDEX_PATH.with_name("faiss.backup.index")
    index = faiss.read_index(str(INDEX_PATH))
    faiss.write_index(index, str(idx_backup))
    print(f" Бэкап: {idx_backup}")

    #6. Дописываем к embeddings.npy
    all_embeddings = np.vstack([embeddings, new_embeddings])
    np.save(EMBEDDINGS_PATH, all_embeddings)
    print(f" Сохранено: {EMBEDDINGS_PATH}")
    print(f"   Было: {n_existing}, стало: {all_embeddings.shape[0]}")

    #7. Дописываем в FAISS
    print(f"   Было в FAISS: {index.ntotal}")
    index.add(new_embeddings.astype(np.float32))
    faiss.write_index(index, str(INDEX_PATH))
    print(f" Сохранено: {INDEX_PATH}")
    print(f"   Стало в FAISS: {index.ntotal}")

    return n_new, all_embeddings.shape

# Main
def main():
    parser = argparse.ArgumentParser(description="Инкрементальные эмбеддинги")
    parser.add_argument(
        "--add", action="store_true",
        help="Добавить эмбеддинги только для новых чанков"
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Пересобрать всё с нуля"
    )
    args = parser.parse_args()

    if not args.add and not args.rebuild:
        print("  Укажите режим: --add или --rebuild")
        print("   python add_embeddings.py --add       (только новые)")
        print("   python add_embeddings.py --rebuild   (всё заново)")
        return

    #Загружаем модель
    print(f" Загружаю модель {EMBED_MODEL_NAME}...")
    t0 = time.time()
    model = SentenceTransformer(EMBED_MODEL_NAME)
    print(f"   Готово за {time.time() - t0:.1f} сек")

    if args.rebuild:
        result = rebuild_all(model)
    else:
        result = add_new(model)

    if result is None:
        return

    #Проверка
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