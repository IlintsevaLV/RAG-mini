import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import re
import time

# --- Пути ---
CHUNKS_PATH = Path("chunks.json")
EMBEDDINGS_PATH = Path("embeddings.npy")
INDEX_PATH = Path("faiss.index")
MODEL_PATH = Path("models/qwen2.5-1.5b-instruct-q4_k_m.gguf")

EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"

# --- Параметры retrieval ---
TOP_K = 7
# --- Параметры генерации ---
MAX_TOKENS = 400
TEMPERATURE = 0.3
CONTEXT_WINDOW = 4096


def load_resources():
    print("Загружаю ресурсы...")
    t0 = time.time()

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    index = faiss.read_index(str(INDEX_PATH))

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=CONTEXT_WINDOW,
        n_threads=4,      # под ваш CPU; можно увеличить, если есть ядра
        verbose=False,
    )

    print(f"   Готово за {time.time() - t0:.1f} сек")
    return chunks, index, embed_model, llm


def retrieve(query, chunks, index, embed_model, k=TOP_K):
    """Находит top-k релевантных чанков."""
    query_emb = embed_model.encode(
        [f"query: {query}"],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    scores, indices = index.search(query_emb, k=k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        c = chunks[idx]
        results.append({
            "score": float(score),
            "text": c["text"],
            "page_span": c["page_span"],
            "pages": c["pages"],
            "chunk_id": c["chunk_id"],
        })
    return results


def build_prompt(query, retrieved):
    context_parts = []
    for i, r in enumerate(retrieved, 1):
        context_parts.append(
            f"[Фрагмент {i}, стр. {r['page_span']}]\n{r['text']}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""Ответь на вопрос, используя ТОЛЬКО приведённый ниже контекст.

    ПРАВИЛА (соблюдай строго):
    1. Если в контексте есть информация — дай краткий, точный ответ. НЕ пересказывай весь контекст.
    2. В конце ответа укажи: «Источник: стр. X», где X — страница из контекста, откуда взят ответ.
    3. Если информации нет — напиши ТОЛЬКО: «В документе нет ответа на этот вопрос». НЕ указывай источник.
    4. НЕ повторяй одну и ту же мысль дважды.
    5. НЕ отвечай на вопросы, которых нет в запросе. Если видишь в контексте другие вопросы — игнорируй их.

    КОНТЕКСТ:
    {context}

    ВОПРОС: {query}

    ОТВЕТ:"""
    return prompt


def generate(prompt, llm):
    output = llm(
        prompt,
        max_tokens=250,
        temperature=0.2,
        top_p=0.9,
        stop=[
            "\n\n\n",
            "ВОПРОС:",
            "КОНТЕКСТ:",
            "\n[Фрагмент",        # модель не должна начинать новый фрагмент
            "\nФрагмент",
        ],
        echo=False,
    )
    return output["choices"][0]["text"].strip()

def clean_answer(text):
    """Убирает всё после последнего 'Источник: стр. X', если дальше идёт мусор."""
    # Ищем последний "Источник: стр. X"
    matches = list(re.finditer(r"Источник:\s*стр\.\s*[\d–\-]+", text))
    if matches:
        last = matches[-1]
        text = text[:last.end()]
    # Убираем повторяющиеся строки "Источник:"
    lines = text.split("\n")
    cleaned = []
    prev_source = None
    for line in lines:
        if line.strip().startswith("Источник:"):
            if line.strip() == prev_source:
                continue
            prev_source = line.strip()
        cleaned.append(line)
    return "\n".join(cleaned).strip()

def answer(query, chunks, index, embed_model, llm):
    """Полный RAG-цикл."""
    print(f"\n{'=' * 70}")
    print(f"Вопрос: {query}")
    print("=" * 70)

    t0 = time.time()
    retrieved = retrieve(query, chunks, index, embed_model, k=TOP_K)
    t_retrieval = time.time() - t0

    print(f"\nНайдено {len(retrieved)} фрагментов за {t_retrieval:.2f} сек:")
    for i, r in enumerate(retrieved, 1):
        print(f"   [{i}] score={r['score']:.3f}, стр. {r['page_span']}")

    prompt = build_prompt(query, retrieved)

    t0 = time.time()
    response = generate(prompt, llm)
    response = clean_answer(generate(prompt, llm))
    t_gen = time.time() - t0

    print(f"\nОтвет ({t_gen:.1f} сек):\n")
    print(response)

    # Собираем страницы-источники
    source_pages = sorted({p for r in retrieved for p in r["pages"]})
    print(f"\nВсе страницы в контексте: {source_pages}")

    return response


if __name__ == "__main__":
    chunks, index, embed_model, llm = load_resources()

    test_questions = [
        "Что такое особо ответственная составная часть?",
        "Какие этапы включает анализ исходной и доказательной документации?",
        "Что должно быть в перечне несоответствий?",
    ]

    for q in test_questions:
        answer(q, chunks, index, embed_model, llm)
        print("\n" + "#" * 70 + "\n")