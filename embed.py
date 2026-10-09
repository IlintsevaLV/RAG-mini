import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
import time

CHUNKS_PATH = Path("chunks.json")
EMBEDDINGS_PATH = Path("embeddings.npy")
INDEX_PATH = Path("faiss.index")

MODEL_NAME = "intfloat/multilingual-e5-large"


def main():
    # --- 1. Загружаем чанки ---
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Загружено чанков: {len(chunks)}")

    # --- 2. Загружаем модель ---
    print(f"Загружаю модель {MODEL_NAME}...")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    print(f"   Загружено за {time.time() - t0:.1f} сек")

    # --- 3. Готовим тексты с префиксом "passage: " ---
    texts = [f"passage: {c['text']}" for c in chunks]

    # --- 4. Считаем эмбеддинги ---
    print(f"Считаю эмбеддинги для {len(texts)} чанков...")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,   # для cosine similarity через inner product
        convert_to_numpy=True,
    )
    print(f"   Посчитано за {time.time() - t0:.1f} сек")
    print(f"   Форма матрицы: {embeddings.shape}")

    # --- 5. Сохраняем эмбеддинги ---
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f" Эмбеддинги сохранены в {EMBEDDINGS_PATH}")

    # --- 6. Строим FAISS-индекс ---
    # Используем IndexFlatIP (inner product) — с нормализованными векторами это = cosine similarity
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    print(f"FAISS-индекс построен: {index.ntotal} векторов, размерность {dim}")

    # --- 7. Сохраняем индекс ---
    faiss.write_index(index, str(INDEX_PATH))
    print(f"FAISS-индекс сохранён в {INDEX_PATH}")

    # --- 8. Быстрый тест: ищем по тестовому запросу ---
    print("\n" + "=" * 70)
    print("ТЕСТ ПОИСКА")
    print("=" * 70)

    test_query = "Что такое особо ответственная составная часть?"
    print(f"\nЗапрос: {test_query}")

    query_emb = model.encode(
        [f"query: {test_query}"],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    scores, indices = index.search(query_emb.astype(np.float32), k=3)

    print("\nТоп-3 результата:")
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        c = chunks[idx]
        print(f"\n--- Ранг {rank} (score={score:.3f}) ---")
        print(f"   pages: {c['page_span']}, chunk_id: {c['chunk_id']}, type: {c['type']}")
        print(f"   Текст: {c['text'][:300]}")
        if len(c['text']) > 300:
            print("   ...")


if __name__ == "__main__":
    main()