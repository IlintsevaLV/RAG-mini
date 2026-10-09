# RAG-mini

Локальная RAG-система (Retrieval-Augmented Generation) для работы с PDF-документами на CPU, без использования GPU. Поддерживает русский язык, OCR сканированных документов и указание страниц-источников в ответах.

## Возможности

- **OCR** сканированных PDF (Tesseract + PyMuPDF)
- **Классификация страниц** — титульник, оглавление, содержание, таблицы
- **Чанкование** с учётом структуры документа (разделы, пункты, подпункты)
- **Эмбеддинги** через `intfloat/multilingual-e5-large`
- **Reranking** через `BAAI/bge-reranker-v2-m3`
- **Гибридный отбор** — top-N по rerank + top-M по embed
- **Генерация ответов** через `Qwen2.5-3B-Instruct` (GGUF, CPU)
- **Ссылки на страницы** в ответах (например, «Источник: стр. 5–6»)

## Требования

- **Python:** 3.12
- **RAM:** минимум 8 ГБ (рекомендуется 16 ГБ)
- **CPU:** 4+ физических ядра
- **Диск:** ~10 ГБ (для моделей)
- **Tesseract OCR:** установлен отдельно (для OCR-этапа)

## Установка

### 1. Клонировать репозиторий

    git clone https://github.com/IlintsevaLV/RAG-mini.git
    cd RAG-mini

### 2. Создать виртуальное окружение

    python -m venv venv

Активировать:

- **Windows:** `venv\Scripts\activate`
- **Linux/macOS:** `source venv/bin/activate`

### 3. Установить зависимости

    pip install -r requirements.txt

### 4. Установить Tesseract OCR

- **Windows:** скачать с https://github.com/UB-Mannheim/tesseract/wiki, отметить русский язык (`rus.traineddata`).
- **Linux:** `sudo apt install tesseract-ocr tesseract-ocr-rus`
- **macOS:** `brew install tesseract tesseract-lang`

### 5. Скачать LLM

Модель `Qwen2.5-3B-Instruct` в GGUF-формате (~2 ГБ):

    python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Qwen/Qwen2.5-3B-Instruct-GGUF', filename='qwen2.5-3b-instruct-q4_k_m.gguf', local_dir='models/')"

### 6. Положить PDF в `data/`

Исходный документ должен лежать в `data/`.

## Запуск пайплайна

### Шаг 1: OCR (если PDF сканированный)

    python ocr_v2.py

Создаёт `ocr_pages_v2.json` — страницы с распознанным текстом.

### Шаг 2: Классификация страниц

    python classify_pages.py

Помечает страницы как `title`, `toc`, `table`, `content`.

### Шаг 3: Чанкование

    python step4_chunking.py

Создаёт `chunks.json` — текстовые чанки с метаданными (`pages`, `page_span`, `type`).

### Шаг 4: Эмбеддинги и FAISS-индекс

    python step5_embed.py

Создаёт `embeddings.npy` и `faiss.index`.

### Шаг 5: RAG-запросы

    python test_questions.py           # пакетный режим (10 вопросов из списка)
    python test_questions.py -i        # интерактивный режим (ввод вручную)
    python test_questions.py -q "..."  # один вопрос

## Структура проекта

    RAG-mini/
    ├── data/                       # исходные PDF
    ├── models/                     # GGUF-модели (не в Git)
    ├── venv/                       # виртуальное окружение (не в Git)
    │
    ├── ocr_v2.py                   # OCR сканированного PDF
    ├── classify_pages.py           # классификация страниц
    ├── step4_chunking.py           # чанкование
    ├── step5_embed.py              # эмбеддинги + FAISS
    ├── test_questions.py           # RAG-запросы
    ├── test_questions_v2.py        # второй набор тестовых вопросов
    │
    ├── ocr_pages_v2.json           # страницы после OCR
    ├── chunks.json                 # чанки
    ├── embeddings.npy              # эмбеддинги (не в Git)
    ├── faiss.index                 # FAISS-индекс (не в Git)
    ├── test_results*.json          # результаты прогонов
    │
    ├── requirements.txt
    ├── .gitignore
    └── README.md

## Архитектура

    PDF → OCR → Классификация → Чанкование → Эмбеддинги → FAISS
                                                                  ↓
    Запрос → Эмбеддинг → FAISS top-N → Reranker → Гибрид → LLM → Ответ

## Параметры (test_questions.py)

| Параметр | Значение | Описание |
|---|---|---|
| `TOP_K_INITIAL` | 15 | Кандидатов из FAISS |
| `K_RERANK` | 2 | Top-N по rerank |
| `K_EMBED` | 3 | +Top-M по embed |
| `TOP_K_FINAL` | 5 | Итого в контексте |
| `MAX_TOKENS` | 500 | Максимум токенов в ответе |
| `TEMPERATURE` | 0.2 | Температура генерации |
| `CONTEXT_WINDOW` | 4096 | Окно контекста LLM |
| `N_THREADS` | 6 | Потоков CPU |

## Ограничения

- **Скорость:** генерация 20–70 сек на ответ (зависит от длины), т.к. LLM работает на CPU.
- **Память:** пиковая нагрузка ~9–10 ГБ RAM (все модели одновременно).
- **Таблицы:** OCR «сплющивает» таблицы, ячейки сливаются. Для структурного распознавания таблиц нужен VLM.
- **Модель:** `Qwen2.5-3B` иногда даёт краткие или неполные ответы на структурные вопросы.

## Лицензия

Проект создан в образовательных целях. Используемые модели имеют свои лицензии:

- `intfloat/multilingual-e5-large` — MIT
- `BAAI/bge-reranker-v2-m3` — MIT
- `Qwen/Qwen2.5-3B-Instruct` — Apache 2.0