import re
import pymorphy2
from collections import defaultdict, Counter

morph = pymorphy2.MorphAnalyzer()

# --- Regex ---
CYR_RE = re.compile(r"^[А-Яа-яЁё]+$")
LAT_RE = re.compile(r"^[A-Za-z]+$")
DIGIT_RE = re.compile(r"^\d+$")

# --- Utils ---
def is_function_word(tok: str) -> bool:
    """
    Проверяем, является ли токен служебным словом
    (предлог, союз, частица).
    """
    if not tok or len(tok) > 3:
        return False
    for p in morph.parse(tok):
        if p.tag.POS in {"PREP", "CONJ", "PRCL"}:
            return True
    return False


def is_cyrillic(tok: str) -> bool:
    """Есть ли кириллические буквы в токене"""
    return bool(re.search(r"[А-Яа-яЁё]", tok))


def is_latin(tok: str) -> bool:
    """Есть ли латиница в токене"""
    return bool(re.search(r"[A-Za-z]", tok))


def token_has_digit(tok: str) -> bool:
    """Есть ли цифра в токене"""
    return any(ch.isdigit() for ch in tok)


def normalize_yo(text: str) -> str:
    """
    Нормализуем 'ё' → 'е' для унификации.
    """
    return text.replace("ё", "е").replace("Ё", "Е")



def collect_adj_for_nouns(df, target_nouns, brand_vocab=None):
    """
    Собираем список прилагательных, которые чаще всего встречаются рядом с целевыми существительными.

    df           : DataFrame (train.csv с колонками sample, annotation)
    target_nouns : множество существительных (например, {"вода", "сок", "молоко"})
    brand_vocab  : словарь брендов (чтобы не спутать прилагательное с брендом)
    """
    if brand_vocab is None:
        brand_vocab = set()

    noun2adjs = defaultdict(Counter)

    for _, row in df.iterrows():
        sample = str(row["sample"])
        try:
            spans = eval(row["annotation"])
        except Exception:
            continue

        toks = [sample[s:e].strip().lower() for s, e, _ in spans]
        labs = [lab for _, _, lab in spans]

        for i, (tok, lab) in enumerate(zip(toks, labs)):
            if tok in target_nouns and lab.endswith("TYPE"):
                neighbors = []
                if i > 0:
                    neighbors.append((toks[i-1], labs[i-1]))
                if i+1 < len(toks):
                    neighbors.append((toks[i+1], labs[i+1]))

                for neigh, neigh_lab in neighbors:
                    if not neigh_lab.endswith("TYPE"):
                        continue
                    # морфология: прилагательное?
                    for p in morph.parse(neigh):
                        if p.tag.POS == "ADJF":
                            if neigh in brand_vocab:
                                continue
                            noun2adjs[tok][normalize_yo(neigh)] += 1
                            break

    # оставляем топ-20 прилагательных для каждого существительного
    return {noun: [adj for adj, _ in cnts.most_common(20)]
            for noun, cnts in noun2adjs.items()}
