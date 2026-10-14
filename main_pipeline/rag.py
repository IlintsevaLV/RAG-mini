import os
os.environ["HF_HUB_OFFLINE"] = "1"

import json
import re
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from llama_cpp import Llama
import time


# Пути
# Пути
BASE_DIR = Path(__file__).resolve().parent      # main_pipeline/
PROJECT_ROOT = BASE_DIR.parent                  # RAG-mini/
CHUNKS_PATH = BASE_DIR / "chunks.json"
INDEX_PATH = BASE_DIR / "faiss.index"
MODEL_PATH = PROJECT_ROOT / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"

EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# Параметры retrieval
TOP_K_INITIAL = 15      # сколько кандидатов берём из FAISS
K_RERANK = 2            # top-N по rerank
K_EMBED = 3             # +top-N по embed
TOP_K_FINAL = K_RERANK + K_EMBED

# Параметры LLM
MAX_TOKENS = 500
TEMPERATURE = 0.1
CONTEXT_WINDOW = 4096
N_THREADS = 6



def load_resources(verbose=True):
    """Загружает chunks, FAISS-индекс, модель эмбеддингов, reranker, LLM."""
    if verbose:
        print("Загружаю ресурсы...")
    t0 = time.time()

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    index = faiss.read_index(str(INDEX_PATH))

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    if verbose:
        print(f"   Загружаю reranker {RERANKER_MODEL_NAME}...")
    reranker = CrossEncoder(RERANKER_MODEL_NAME)

    if verbose:
        print(f"   Загружаю LLM {MODEL_PATH.name}...")
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=CONTEXT_WINDOW,
        n_threads=N_THREADS,
        verbose=False,
    )

    if verbose:
        print(f"   Готово за {time.time() - t0:.1f} сек\n")

    return chunks, index, embed_model, reranker, llm

# Retrieval
def retrieve_initial(query, chunks, index, embed_model, k=TOP_K_INITIAL):
    """FAISS ищет k кандидатов по эмбеддингам."""
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
            "score_embed": float(score),
            "text": c["text"],
            "page_span": c["page_span"],
            "pages": c["pages"],
            "chunk_id": c["chunk_id"],
            "source": c.get("source", "?"),
        })
    return results


def rerank_candidates(query, candidates, reranker):
    """Считает rerank-скоры для всех кандидатов."""
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    for c, s in zip(candidates, scores):
        c["score_rerank"] = float(s)
    return candidates


def hybrid_select(candidates, k_rerank=K_RERANK, k_embed=K_EMBED):
    """top-k_rerank по rerank + top-k_embed по embed (без пересечений)."""
    by_rerank = sorted(candidates, key=lambda c: c["score_rerank"], reverse=True)
    selected = by_rerank[:k_rerank]
    selected_ids = {c["chunk_id"] for c in selected}

    by_embed = sorted(candidates, key=lambda c: c["score_embed"], reverse=True)
    for c in by_embed:
        if len(selected) >= k_rerank + k_embed:
            break
        if c["chunk_id"] not in selected_ids:
            selected.append(c)
            selected_ids.add(c["chunk_id"])

    return selected


def retrieve(query, chunks, index, embed_model, reranker):
    """Полный retrieval: FAISS top-N → rerank → гибридный отбор."""
    candidates = retrieve_initial(query, chunks, index, embed_model, k=TOP_K_INITIAL)
    candidates = rerank_candidates(query, candidates, reranker)
    return hybrid_select(candidates, k_rerank=K_RERANK, k_embed=K_EMBED)

# Промпт и генерация
def build_prompt(query, retrieved):
    context_parts = []
    for i, r in enumerate(retrieved, 1):
        source = r.get("source", "?")
        context_parts.append(
             f"[Фрагмент {i}, документ: «{source}», стр. {r['page_span']}]\n{r['text']}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""Ответь на вопрос, используя ТОЛЬКО приведённый ниже контекст.

ПРАВИЛА (соблюдай строго):
1. Дай краткий ответ на вопрос. Если вопрос просит перечислить — перечисли ВСЕ пункты из контекста.
2. Используй ТОЛЬКО текст из контекста.
3. Если в контексте нет прямого ответа — напиши «В документе нет ответа на этот вопрос».
4. В конце ответа укажи:  «Источник: <название документа>, стр. X».

КОНТЕКСТ:
{context}

ВОПРОС: {query}

ОТВЕТ:"""
    return prompt


def generate(prompt, llm):
    output = llm(
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=0.9,
        stop=[
            "\n\n\n",
            "\nВОПРОС:",
            "\nКОНТЕКСТ:",
            "\n[Фрагмент",
            "\nФрагмент",
            "\nВопрос",
            "\nОтвет",
        ],
        echo=False,
    )
    return output["choices"][0]["text"].strip()


def clean_answer(text):
    """Постобработка ответа."""
    # 1. Обрезаем "Вопрос:"
    match = re.search(r"(?:^|\n)\s*Вопрос\s*:", text, re.MULTILINE)
    if match and match.start() > 20:
        text = text[:match.start()]

    # 2. Обрезаем после первой ссылки на источник
    match = re.search(r"Источник:\s*.+?стр\.\s*[\d–\-]+", text)
    if match:
        text = text[:match.end()]
    else:
        for marker in [
            "В контексте", "Согласно контексту", "Из контекста",
            "Таким образом", "Следовательно", "Вот список",
            "Приведены следующие пункты", "Пример ответа",
            "Это соответствует", "Следует отметить",
        ]:
            idx = text.find(marker)
            if idx > 50:
                text = text[:idx]
                break

    # 3. Убираем фразы-мусор
    if len(text) > 100:
        cleaned = re.sub(r"\s*ТОЛЬКО[^\n]*", "", text)
        cleaned = re.sub(r"\s*Это соответствует пункту[^\n]*", "", cleaned)
        cleaned = re.sub(r"\s*\d+\.\s+(?=\d+\.)", "", cleaned)
        cleaned = re.sub(r"\s*\d+\.\s*$", "", cleaned)

        without_no_answer = re.sub(
            r"\s*В документе нет ответа на этот вопрос\.?", "", cleaned
        ).strip()

        is_only_source = bool(re.fullmatch(
            r"Источник:\s*.+?стр\.\s*[\d–\-]+\.?", without_no_answer
        ))

        if len(without_no_answer) > 100 and not is_only_source:
            cleaned = without_no_answer

        text = cleaned

    # 4. Убираем повторяющиеся фразы
    for sep in [";", ". ", ", ", " – ", " - "]:
        parts = text.split(sep)
        seen = set()
        unique = []
        for p in parts:
            key = p.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(p)
        text = sep.join(unique)

    # 5. Убираем повторяющиеся предложения
    sentences = re.split(r"(?<=[.!?])\s+", text)
    seen = set()
    unique = []
    for s in sentences:
        key = s.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(s)
    text = " ".join(unique).strip()

    # ============================================================
    # 6. ГЛАВНОЕ: если ответ содержит "нет ответа" — возвращаем ТОЛЬКО эту фразу
    # ============================================================
    if "нет ответа на этот вопрос" in text.lower():
        return "В документе нет ответа на этот вопрос"

    # 7. Если остался ТОЛЬКО источник — тоже возвращаем "нет ответа"
    if re.fullmatch(r"\s*Источник:\s*.+?стр\.\s*[\d–\-]+\.?\s*", text):
        return "В документе нет ответа на этот вопрос"

    return text.strip()


# Основной RAG
def answer_one(query, chunks, index, embed_model, reranker, llm, verbose=True):
    """Полный RAG-цикл для одного вопроса."""
    t0 = time.time()
    retrieved = retrieve(query, chunks, index, embed_model, reranker)
    t_ret = time.time() - t0

    if verbose:
        print(f"\nТоп-{TOP_K_FINAL} (за {t_ret:.2f} сек):")
        for i, r in enumerate(retrieved, 1):
            print(f"   [{i}] embed={r['score_embed']:.3f}, "
                  f"rerank={r['score_rerank']:.3f}, стр. {r['page_span']}")

    prompt = build_prompt(query, retrieved)

    t0 = time.time()
    raw_response = generate(prompt, llm)
    response = clean_answer(raw_response)
    t_gen = time.time() - t0

    if verbose:
        print(f"\nОтвет ({t_gen:.1f} сек):\n{response}")

    return {
        "question": query,
        "retrieved": retrieved,
        "raw_answer": raw_response,
        "answer": response,
        "time_retrieval": t_ret,
        "time_generation": t_gen,
    }