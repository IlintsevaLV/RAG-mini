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
from rank_bm25 import BM25Okapi


# Пути
BASE_DIR = Path(__file__).resolve().parent      # main_pipeline/
PROJECT_ROOT = BASE_DIR.parent                  # RAG-mini/
CHUNKS_PATH = BASE_DIR / "chunks.json"
INDEX_PATH = BASE_DIR / "faiss.index"
MODEL_PATH = PROJECT_ROOT / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"

EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# Параметры retrieval
TOP_K_INITIAL = 20       # FAISS top-N
TOP_K_BM25 = 20          # BM25 top-N
RRF_K = 60               # константа сглаживания RRF
RRF_TOP_N = 20           # сколько чанков оставляем после RRF (идут в reranker)
TOP_K_FINAL = 6          # сколько чанков идёт в LLM после reranker'а

# Параметры расширения таблиц
MAX_ADDED_PARTS = 3            # добавить не больше 3 новых частей таблицы
MAX_ADDED_CHARS = 6000         # добавить не больше 6000 новых символов
MIN_RERANK_TO_EXPAND = 0.5     # порог: расширяем только если rerank таблицы выше

# Параметры LLM
MAX_TOKENS = 500
TEMPERATURE = 0.1
CONTEXT_WINDOW = 4096
N_THREADS = 6



def load_resources(verbose=True):
    """Загружает chunks, FAISS-индекс, BM25-индекс, embed-модель, reranker, LLM."""
    if verbose:
        print("Загружаю ресурсы...")
    t0 = time.time()

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    index = faiss.read_index(str(INDEX_PATH))

    if verbose:
        print("   Строю BM25-индекс...")
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)

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

    return chunks, index, embed_model, reranker, llm, bm25

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
            "type": c.get("type", "content"),
            "table_caption": c.get("table_caption", ""),
            "table_group": c.get("table_group", None),
            "table_part": c.get("table_part", None),
        })
    return results

def retrieve_bm25(query, chunks, bm25, k=TOP_K_BM25):
    """BM25-поиск: ловит точные совпадения слов/чисел/аббревиатур.

    Возвращает список чанков с полем score_bm25.
    """
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)

    # Индексы top-k по убыванию
    top_idx = np.argsort(scores)[::-1][:k]

    results = []
    for idx in top_idx:
        c = chunks[idx]
        results.append({
            "score_bm25": float(scores[idx]),
            "text": c["text"],
            "page_span": c["page_span"],
            "pages": c["pages"],
            "chunk_id": c["chunk_id"],
            "source": c.get("source", "?"),
            "type": c.get("type", "content"),
            "table_caption": c.get("table_caption", ""),
            "table_group": c.get("table_group", None),
            "table_part": c.get("table_part", None),
        })
    return results

def rrf_fusion(rankings, k=RRF_K, top_n=RRF_TOP_N):
    """Reciprocal Rank Fusion: объединяет несколько ранжирований в одно.

    Args:
        rankings: список ранжирований. Каждое — список чанков (dict),
                  упорядоченных по релевантности (лучший — первый).
        k: константа сглаживания RRF (обычно 60).
        top_n: сколько чанков вернуть.

    Returns:
        Список чанков, упорядоченных по RRF-скору.

    Формула:
        RRF_score(chunk) = Σ_i 1 / (k + rank_i(chunk))
    """
    scores = {}
    lookup = {}

    for ranking in rankings:
        for rank, c in enumerate(ranking, 1):
            cid = c["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            # Сохраняем первый попавшийся объект чанка (можно любой)
            if cid not in lookup:
                lookup[cid] = c

    sorted_ids = sorted(scores.keys(), key=lambda cid: -scores[cid])
    result = []
    for cid in sorted_ids[:top_n]:
        chunk = lookup[cid].copy()
        chunk["score_rrf"] = scores[cid]
        result.append(chunk)
    return result

def rerank_candidates(query, candidates, reranker):
    """Считает rerank-скоры для всех кандидатов."""
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    for c, s in zip(candidates, scores):
        c["score_rerank"] = float(s)
    return candidates


def expand_table_neighbors(selected, chunks, min_rerank_score=MIN_RERANK_TO_EXPAND):
    """Подтягивает соседние части ОДНОЙ таблицы, если её rerank выше порога.

    Логика:
        - Находит табличные чанки в selected с их table_group.
        - Берёт группу с максимальным score_rerank.
        - Если score_rerank < min_rerank_score → не расширяем (таблица нерелевантна).
        - Иначе дотягивает недостающие части той же таблицы из chunks.
        - Лимиты: не больше MAX_ADDED_PARTS частей и MAX_ADDED_CHARS символов.
    """
    selected_ids = {c["chunk_id"] for c in selected}

    # 1. Табличные чанки в selected
    tables_in_selected = [
        (c, c.get("table_group"))
        for c in selected
        if c.get("table_group")
    ]
    if not tables_in_selected:
        return selected

    # 2. Берём самую релевантную таблицу
    tables_in_selected.sort(
        key=lambda x: x[0].get("score_rerank", 0), reverse=True
    )
    best_chunk, best_group = tables_in_selected[0]

    # 3. Фильтр по скору
    if best_chunk.get("score_rerank", 0) < min_rerank_score:
        return selected

    # 4. Все части этой таблицы
    all_parts = [c for c in chunks if c.get("table_group") == best_group]
    if not all_parts:
        return selected

    all_parts_sorted = sorted(all_parts, key=lambda c: c.get("table_part", 0))

    # Уже в контексте
    already = [c for c in all_parts_sorted if c["chunk_id"] in selected_ids]

    # Кандидаты на добавление
    candidates = [c for c in all_parts_sorted if c["chunk_id"] not in selected_ids]

    # Сортируем: сначала ближайшие по table_part к уже выбранным
    def distance_to_selected(c):
        part = c.get("table_part", 0)
        return min(abs(part - sel.get("table_part", 0)) for sel in already)

    candidates.sort(key=distance_to_selected)

    # 5. Добавляем с лимитами
    expanded = list(selected)
    added_count = 0
    added_chars = 0

    for c in candidates:
        if added_count >= MAX_ADDED_PARTS:
            break
        if added_chars + c["n_chars"] > MAX_ADDED_CHARS:
            break
        expanded.append(c)
        added_count += 1
        added_chars += c["n_chars"]

    return expanded

def retrieve(query, chunks, index, embed_model, reranker, bm25):
    """Гибридный retrieval:
        1. FAISS top-N (семантика)
        2. BM25 top-N (ключевые слова)
        3. RRF: объединение двух ранжирований
        4. Reranker: точная пересортировка top-RRF_TOP_N
        5. Top-TOP_K_FINAL по reranker'у
        6. expand_table_neighbors — дотягиваем соседние части таблиц
    """
    # 1. FAISS
    faiss_res = retrieve_initial(query, chunks, index, embed_model, k=TOP_K_INITIAL)

    # 2. BM25
    bm25_res = retrieve_bm25(query, chunks, bm25, k=TOP_K_BM25)

    # 3. RRF
    fused = rrf_fusion([faiss_res, bm25_res], k=RRF_K, top_n=RRF_TOP_N)

    # 4. Reranker по объединённому пулу
    fused = rerank_candidates(query, fused, reranker)

    # 5. Top-K по reranker'у
    selected = sorted(fused, key=lambda c: c["score_rerank"], reverse=True)[:TOP_K_FINAL]

    # 6. Расширение таблиц
    n_before = len(selected)
    selected = expand_table_neighbors(selected, chunks)
    n_after = len(selected)

    if n_after > n_before:
        print(f"   [expand_table_neighbors: {n_before} → {n_after} чанков]")

    return selected

# Промпт и генерация
def build_prompt(query, retrieved):
    context_parts = []
    for i, r in enumerate(retrieved, 1):
        source = r.get("source", "?")
        if r.get("type") == "table":
            caption = r.get("table_caption") or "(без заголовка)"
            context_parts.append(
                f"[Фрагмент {i}, ТАБЛИЦА «{caption}», документ: «{source}», стр. {r['page_span']}]\n{r['text']}"
            )
        else:
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

    # 6. ГЛАВНОЕ: если ответ содержит "нет ответа" — возвращаем ТОЛЬКО эту фразу

    if "нет ответа на этот вопрос" in text.lower():
        return "В документе нет ответа на этот вопрос"

    # 7. Если остался ТОЛЬКО источник — тоже возвращаем "нет ответа"
    if re.fullmatch(r"\s*Источник:\s*.+?стр\.\s*[\d–\-]+\.?\s*", text):
        return "В документе нет ответа на этот вопрос"

    return text.strip()


# Основной RAG
def answer_one(query, chunks, index, embed_model, reranker, llm, bm25, verbose=True):
    """Полный RAG-цикл для одного вопроса."""
    t0 = time.time()
    retrieved = retrieve(query, chunks, index, embed_model, reranker, bm25)
    t_ret = time.time() - t0

    if verbose:
        print(f"\nТоп-{len(retrieved)} (за {t_ret:.2f} сек):")
        for i, r in enumerate(retrieved, 1):
            type_marker = "TABLE" if r.get("type") == "table" else "text "
            caption_info = ""
            if r.get("type") == "table" and r.get("table_caption"):
                caption_info = f", «{r['table_caption']}»"

            # Новые чанки (из expand_table_neighbors) не имеют score_embed/score_rerank
            embed_val = r.get("score_embed")
            rerank_val = r.get("score_rerank")
            embed_str = f"{embed_val:.3f}" if embed_val is not None else "  —  "
            rerank_str = f"{rerank_val:.3f}" if rerank_val is not None else "  —  "

            print(f"   [{i}] {type_marker} embed={embed_str}, "
                  f"rerank={rerank_str}, стр. {r['page_span']}{caption_info}")

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