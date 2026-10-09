import json
import re
from pathlib import Path
from collections import Counter

INPUT_PATH = Path("ocr_pages_v2.json")
OUTPUT_PATH = Path("ocr_pages_v2.json")


# ---------- Метрики ----------
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
    return sum(1 for l in ls if re.search(r"\.{3,}", l)) / len(ls)


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


# ---------- Классификатор ----------
def classify_page(page: dict) -> str:
    text = page["text"]
    n = n_chars(text)
    ls = lines(text)
    if not ls:
        return "content"

    # 1. Титульник
    title_markers = ["МЕЖГОСУДАРСТВЕННЫЙ", "АВИАЦИОННЫЙ КОМИТЕТ", "АВИАЦИОННЫЙ РЕГИСТР"]
    if n < 500 and any(m in text for m in title_markers):
        return "title"

    # 2. Таблица (проверяем РАНЬШЕ оглавления)
    frac_pipe = frac_pipe_chars(text)
    many_spaces = len(re.findall(r" {3,}", text))
    n_slash = len(re.findall(r"\bn/n\b", text, flags=re.IGNORECASE))
    n_num = len(re.findall(r"№", text))
    is_table = (
        frac_pipe > 0.02
        or many_spaces > 15
        or n_slash > 3
        or n_num > 3
    )
    if is_table:
        return "table"

    # 3. Оглавление
    frac_digit = frac_lines_ending_with_digit(text)
    frac_dots = frac_lines_with_dots(text)
    frac_punct = frac_dots_and_commas(text)
    n_slash = len(re.findall(r"\bn/n\b", text, flags=re.IGNORECASE))

    # Главный признак оглавления: много строк С отточиями И с номером страницы
    is_toc = (
            frac_dots > 0.3 and frac_digit > 0.3 and frac_pipe < 0.01 and n_slash == 0
    )

    # Явный заголовок "СОДЕРЖАНИЕ" в первых строках + отточия ниже
    if not is_toc:
        first_lines = [l.upper() for l in ls[:3]]
        has_header = any(
            re.fullmatch(r"СОДЕРЖАНИЕ[\s.:]*", l) or re.fullmatch(r"ОГЛАВЛЕНИЕ[\s.:]*", l)
            for l in first_lines
        )
        if has_header and frac_dots > 0.2:
            is_toc = True

    if is_toc:
        return "toc"

    return "content"


# ---------- Прогон ----------
if __name__ == "__main__":
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        pages = json.load(f)

    print(f"Классифицирую {len(pages)} страниц\n")
    print(f"{'стр.':>5} {'симв.':>6}  {'тип':<9} метрики")
    print("-" * 100)

    for p in pages:
        p["type"] = classify_page(p)
        print(
            f"{p['page']:>5} {p['n_chars']:>6}  {p['type']:<9} "
            f"| dgt={frac_lines_ending_with_digit(p['text']):.2f} "
            f"dots={frac_lines_with_dots(p['text']):.2f} "
            f"len={avg_line_length(p['text']):.1f} "
            f"punct={frac_dots_and_commas(p['text']):.4f} "
            f"pipe={frac_pipe_chars(p['text']):.4f} "
            f"sp={len(re.findall(r' {3,}', p['text']))} "
            f"n/n={len(re.findall(r'\\bn/n\\b', p['text'], flags=re.IGNORECASE))} "
            f"num={len(re.findall(r'№', p['text']))}"
        )

    c = Counter(p["type"] for p in pages)
    print("\nИтог по типам:")
    for t, cnt in c.most_common():
        print(f"  {t:<8} {cnt}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    print(f"\n Обновлено: {OUTPUT_PATH}")