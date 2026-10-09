import os
os.environ["HF_HUB_OFFLINE"] = "1"
import json
import argparse
import re
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from llama_cpp import Llama
import time

import sys
sys.stdout.reconfigure(line_buffering=True)


# --- Пути ---
CHUNKS_PATH = Path("chunks.json")
INDEX_PATH = Path("faiss.index")
MODEL_PATH = Path("models/qwen2.5-3b-instruct-q4_k_m.gguf")
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# --- Параметры ---
TOP_K_INITIAL = 15      # сколько кандидатов берём из FAISS
TOP_K_FINAL = 5         # сколько оставляем после reranking
MAX_TOKENS = 500
TEMPERATURE = 0.1
CONTEXT_WINDOW = 4096
N_THREADS = 6

QUESTIONS = [
    "Что такое авиационная техника?",
    "Что такое доказательная документация?",
    "Какие объекты подлежат проверке при оценке соответствия технологии?",
    "Что такое модификация?",
    "Какие требования предъявляются к актам внедрения ООТП?",
    "Что такое сертификационные заводские испытания?",
    "Какие документы хранятся в деле предприятия и сколько лет?",
    "Что такое специальный процесс?",
    "Какие этапы включают дополнительные сертификационные работы?",
    "Кто допускается к участию в сертификации в качестве экспертов?",
]
'''
QUESTIONS = [
    "Что такое сертификат типа?",
    "Что такое типовая конструкция?",
    "Какие классы несоответствий бывают и чем они отличаются?",
    "Что входит в перечень исходной доказательной документации?",
    "Какие функции выполняют сертификационные центры по п. 605?",
    "Что такое особо ответственный технологический процесс?",
    "Сколько этапов в анализе ИиДД и какие?",
    "Что должно быть в протоколе оценки типовой конструкции?",
    "Какие требования к технологии изготовления устанавливает п. 605?",
    "Что такое литирование документации?",
]
'''

def load_resources():
    print(" Загружаю ресурсы...")
    t0 = time.time()
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    index = faiss.read_index(str(INDEX_PATH))
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    print(f"   Загружаю reranker {RERANKER_MODEL_NAME}...")
    reranker = CrossEncoder(RERANKER_MODEL_NAME)
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=CONTEXT_WINDOW,
        n_threads=N_THREADS,
        verbose=False,
    )
    print(f"   Готово за {time.time() - t0:.1f} сек\n")
    return chunks, index, embed_model, reranker, llm


def retrieve_initial(query, chunks, index, embed_model, k=TOP_K_INITIAL):
    """Первый этап: FAISS ищет k кандидатов по эмбеддингам."""
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
        })
    return results


def rerank(query, candidates, reranker, k_final=TOP_K_FINAL):
    """Второй этап: cross-encoder пересортировывает кандидатов."""
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    for c, s in zip(candidates, scores):
        c["score_rerank"] = float(s)
    reranked = sorted(candidates, key=lambda c: c["score_rerank"], reverse=True)
    return reranked[:k_final]


def retrieve(query, chunks, index, embed_model, reranker):
    """FAISS top-N → rerank all → hybrid select."""
    candidates = retrieve_initial(query, chunks, index, embed_model, k=TOP_K_INITIAL)

    # Считаем rerank-скоры для ВСЕХ кандидатов
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    for c, s in zip(candidates, scores):
        c["score_rerank"] = float(s)

    # Гибридный отбор: 2 rerank + 3 embed
    return hybrid_select(candidates, k_rerank=2, k_embed=3)


def hybrid_select(candidates, k_rerank=2, k_embed=3):
    """top-2 по rerank + top-3 по embed (без пересечений)."""
    # 1. Сортируем по rerank
    by_rerank = sorted(candidates, key=lambda c: c["score_rerank"], reverse=True)
    selected = by_rerank[:k_rerank]
    selected_ids = {c["chunk_id"] for c in selected}

    # 2. Добавляем top по embed, которых ещё нет
    by_embed = sorted(candidates, key=lambda c: c["score_embed"], reverse=True)
    for c in by_embed:
        if c["chunk_id"] not in selected_ids:
            selected.append(c)
            selected_ids.add(c["chunk_id"])
        if len(selected) >= k_rerank + k_embed:
            break

    return selected

def build_prompt(query, retrieved):
    context_parts = []
    for i, r in enumerate(retrieved, 1):
        context_parts.append(
            f"[Фрагмент {i}, стр. {r['page_span']}]\n{r['text']}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""Ответь на вопрос, используя ТОЛЬКО приведённый ниже контекст.

ПРАВИЛА (соблюдай строго):
1. Дай краткий ответ на вопрос. Если вопрос просит перечислить — перечисли ВСЕ пункты из контекста.
2. Используй ТОЛЬКО текст из контекста.
3. Если в контексте нет прямого ответа — напиши «В документе нет ответа на этот вопрос».
4. В конце ответа укажи: «Источник: стр. X».

КОНТЕКСТ:
{context}

ВОПРОС: {query}

ОТВЕТ:"""
    return prompt


def clean_answer(text):
    """Обрезает после первой ссылки, убирает повторяющиеся предложения."""
    # 1. Обрезаем всё после "Вопрос:" / "Ответ:"
    match = re.search(r"(?:^|\n)\s*(?:Вопрос|Ответ)\s*:", text, re.MULTILINE)
    if match:
        text = text[:match.start()]

    # 2. Обрезаем после первой ссылки на источник
    match = re.search(r"Источник:\s*стр\.\s*[\d–\-]+", text)
    if match:
        text = text[:match.end()]
    else:
        # 2b. Если "Источник" нет — обрезаем по маркерам рассуждений
        for marker in ["В контексте", "Согласно контексту", "Из контекста",
                       "Таким образом", "Следовательно", "Вот список",
                       "Приведены следующие пункты", "Пример ответа",
                       "Это соответствует", "Следует отметить"]:
            idx = text.find(marker)
            if idx > 50:   # не режем, если маркер в самом начале
                text = text[:idx]
                break

    # 3. Убираем фразы-мусор
    if len(text) > 100:
        text = re.sub(r"\s*В документе нет ответа на этот вопрос\.?", "", text)
        text = re.sub(r"\s*ТОЛЬКО[^\n]*", "", text)
        text = re.sub(r"\s*Это соответствует пункту[^\n]*", "", text)
        text = re.sub(r"\s*\d+\.\s+(?=\d+\.)", "", text)  # "1. 2. 3." → мусор
        text = re.sub(r"\s*\d+\.\s*$", "", text)  # висячий "5."

    # 4. Убираем повторяющиеся ПРЕДЛОЖЕНИЯ
    sentences = re.split(r"(?<=[.!?])\s+", text)
    seen = set()
    unique = []
    for s in sentences:
        s_key = s.strip().lower()
        if s_key and s_key not in seen:
            seen.add(s_key)
            unique.append(s)
    text = " ".join(unique)

    return text.strip()


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


def answer_one(query, chunks, index, embed_model, reranker, llm, verbose=True):
    t0 = time.time()
    retrieved = retrieve(query, chunks, index, embed_model, reranker)
    t_ret = time.time() - t0

    if verbose:
        print(f"\n Топ-{TOP_K_FINAL} после reranking (за {t_ret:.2f} сек):")
        for i, r in enumerate(retrieved, 1):
            print(f"   [{i}] embed={r['score_embed']:.3f}, rerank={r['score_rerank']:.3f}, стр. {r['page_span']}")

    prompt = build_prompt(query, retrieved)
    t0 = time.time()
    raw_response = generate(prompt, llm)
    response = clean_answer(raw_response)
    t_gen = time.time() - t0

    if verbose:
        print(f"\n Ответ ({t_gen:.1f} сек):\n{response}")

    return {
        "question": query,
        "retrieved": retrieved,
        "raw_answer": raw_response,
        "answer": response,
        "time_retrieval": t_ret,
        "time_generation": t_gen,
    }


def run_batch(chunks, index, embed_model, reranker, llm):
    results_log = []
    for i, q in enumerate(QUESTIONS, 1):
        print("\n" + "=" * 70)
        print(f"[{i}/{len(QUESTIONS)}]  {q}")
        print("=" * 70)
        r = answer_one(q, chunks, index, embed_model, reranker, llm)
        results_log.append(r)

    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(results_log, f, ensure_ascii=False, indent=2)
    print(f"\n Результаты сохранены в test_results.json")


def run_interactive(chunks, index, embed_model, reranker, llm):
    print("\n" + "=" * 70)
    print(" ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print("Введите вопрос или 'q' для выхода.")
    print("=" * 70)
    while True:
        try:
            query = input("\n Ваш вопрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break
        if query.lower() in ("q", "exit", "quit", ""):
            print("Выход.")
            break
        print("=" * 70)
        answer_one(query, chunks, index, embed_model, reranker, llm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--question", "-q", type=str, default=None)
    args = parser.parse_args()

    chunks, index, embed_model, reranker, llm = load_resources()

    if args.question:
        print("=" * 70)
        print(f" {args.question}")
        print("=" * 70)
        answer_one(args.question, chunks, index, embed_model, reranker, llm)
    elif args.interactive:
        run_interactive(chunks, index, embed_model, reranker, llm)
    else:
        run_batch(chunks, index, embed_model, reranker, llm)