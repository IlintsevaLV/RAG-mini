"""
Классификация страниц в pages.json.

Режимы:
    python classify_pages.py        # классифицировать ВСЕ (перезапись)
    python classify_pages.py --add  # только документы БЕЗ поля 'type'

Классификация добавляет каждой странице поле "type":
  - "title"   — титульник
  - "toc"     — оглавление / содержание
  - "table"   — таблица / форма
  - "content" — обычный текст
"""

import argparse
import json
import re
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent
PAGES_PATH = BASE_DIR / "pages.json"

# Метрики
def lines(text):
    return [l.strip() for l in text.splitlines() if l.strip()]


def frac_lines_ending_with_digit(text):
    ls = lines(text)
    if not ls:
        return 0.0
    return sum(1 for l in ls if re.search(r"\d+\s*$", l)) / len(ls)


def frac_lines_with_dots(text):
    ls = lines(text)
    if not ls:
        return 0.0
    # Ищем 3+ точек, возможно разделённых пробелами:
    # "..." или ". . ." или ". . . ."
    pattern = re.compile(r"(\.[\s]?){3,}")
    return sum(1 for l in ls if pattern.search(l)) / len(ls)


def avg_line_length(text):
    ls = lines(text)
    if not ls:
        return 0.0
    return sum(len(l) for l in ls) / len(ls)


def frac_dots_and_commas(text):
    if not text:
        return 0.0
    punct = sum(1 for ch in text if ch in ".,;:!?")
    return punct / len(text)


def frac_pipe_chars(text):
    if not text:
        return 0.0
    return text.count("|") / len(text)


def n_chars(text):
    return len(text.strip())



# Классификатор
def classify_page(page: dict) -> str:
    text = page["text"]
    page_num = page["page"]
    n = n_chars(text)
    ls = lines(text)
    if not ls:
        return "content"

    has_tables = page.get("n_tables", 0) > 0
    text_upper = text.upper()

    # 1. ТИТУЛЬНИК — строго: только стр. 1–3 + сильные признаки
    if page_num <= 3 and n < 1500:
        doc_type_markers = [
            "РЕКОМЕНДАТЕЛЬНЫЙ ЦИРКУЛЯР",
            "ДИРЕКТИВНОЕ ПИСЬМО",
            "РУКОВОДСТВО 23-",
            "РУКОВОДСТВО 24-",
            "РУКОВОДСТВО 25-",
            "МЕТОДЫ ОПРЕДЕЛЕНИЯ СООТВЕТСТВИЯ",
            "МЕТОДЫ ОБЕСПЕЧЕНИЯ СООТВЕТСТВИЯ",
            "НОРМА ЛЕТНОЙ ГОДНОСТИ",
            "АВИАЦИОННЫЕ ПРАВИЛА, ЧАСТЬ",
        ]
        intro_markers = [
            "ВВЕДЕН В ДЕЙСТВИЕ",
            "ВВЕДЕНО В ДЕЙСТВИЕ",
            "УТВЕРЖДЕН",
            "УТВЕРЖДЕНО",
            "ОДОБРЕН",
            "ОДОБРЕНО",
            "РЕШЕНИЕМ ПРОТОКОЛА",
        ]

        has_type = any(m in text_upper for m in doc_type_markers)
        has_intro = any(m in text_upper for m in intro_markers)

        last_lines = " ".join(ls[-3:])
        has_year = bool(re.search(r"\b(19|20)\d{2}\s*г?\.?\b", last_lines))

        if has_type and (has_intro or has_year):
            return "title"

    # 2. ТАБЛИЦА
    frac_pipe = frac_pipe_chars(text)
    many_spaces = len(re.findall(r" {3,}", text))
    n_slash = len(re.findall(r"\bn/n\b", text, flags=re.IGNORECASE))
    n_num = len(re.findall(r"№", text))

    if (frac_pipe > 0.02 or many_spaces > 15 or n_slash > 3 or n_num > 3):
        return "table"

    # 3. ОГЛАВЛЕНИЕ
    frac_dots = frac_lines_with_dots(text)
    frac_digit = frac_lines_ending_with_digit(text)

    is_toc = (
        frac_dots > 0.3 and frac_digit > 0.3 and frac_pipe < 0.01 and n_slash == 0
    )

    if not is_toc:
        first_lines = [l.upper() for l in ls[:3]]
        has_header = any(
            re.fullmatch(r"СОДЕРЖАНИЕ[\s.:]*", l)
            or re.fullmatch(r"ОГЛАВЛЕНИЕ[\s.:]*", l)
            or l.startswith("СОДЕРЖАНИЕ .")
            for l in first_lines
        )
        if has_header and frac_dots > 0.2:
            is_toc = True

    if is_toc:
        return "toc"

    # 4. CONTENT
    return "content"



def needs_classification(doc):
    """True, если хотя бы у одной страницы нет поля 'type'."""
    return any("type" not in p for p in doc["pages"])


def classify_document(doc, doc_name):
    """Классифицирует все страницы одного документа"""
    print(f"\n {doc_name}")
    print(f"{'стр.':>5} {'симв.':>6} {'tbl':>4}  {'тип':<9} метрики")
    print("-" * 100)

    for p in doc["pages"]:
        p["type"] = classify_page(p)
        print(
            f"{p['page']:>5} {p['n_chars']:>6} {p.get('n_tables', 0):>4}  {p['type']:<9} "
            f"| dgt={frac_lines_ending_with_digit(p['text']):.2f} "
            f"dots={frac_lines_with_dots(p['text']):.2f} "
            f"len={avg_line_length(p['text']):.1f} "
            f"punct={frac_dots_and_commas(p['text']):.4f} "
            f"pipe={frac_pipe_chars(p['text']):.4f} "
            f"sp={len(re.findall(r' {3,}', p['text']))} "
            f"n/n={len(re.findall(r'\\bn/n\\b', p['text'], flags=re.IGNORECASE))} "
            f"num={len(re.findall(r'№', p['text']))}"
        )


def main():
    parser = argparse.ArgumentParser(description="Классификация страниц в pages.json")
    parser.add_argument(
        "--add", action="store_true",
        help="Классифицировать только документы без поля 'type'"
    )
    args = parser.parse_args()

    if not PAGES_PATH.exists():
        print(f" Файл {PAGES_PATH} не найден.")
        return

    with open(PAGES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_pages = sum(len(doc["pages"]) for doc in data.values())
    print(f" Документов в pages.json: {len(data)}")
    print(f" Всего страниц: {total_pages}")

    # Решаем, что классифицировать -
    if args.add:
        targets = [name for name, doc in data.items() if needs_classification(doc)]
        if not targets:
            print("\n Все документы уже классифицированы. Нечего делать.")
            return
        print(f"\n Найдено необработанных документов: {len(targets)}")
        for name in targets:
            print(f"   • {name}")
    else:
        targets = list(data.keys())
        print(f"\n Классифицирую ВСЕ документы ({len(targets)})")

    #Бэкап ДО изменений
    backup_path = PAGES_PATH.with_name("pages.backup.json")
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f" Бэкап: {backup_path}")

    # --- Классифицируем ---
    for doc_name in targets:
        classify_document(data[doc_name], doc_name)
        print()

    # --- Сохраняем ---
    with open(PAGES_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f" Обновлено: {PAGES_PATH}")

    # --- Сводка ---
    all_types = [p["type"] for doc in data.values() for p in doc["pages"]]
    print(f"\n Итог по типам (по всем документам):")
    for t, cnt in Counter(all_types).most_common():
        print(f"  {t:<8} {cnt}")


if __name__ == "__main__":
    main()