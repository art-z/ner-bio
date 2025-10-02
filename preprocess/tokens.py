import re
from .utils import normalize_yo

# --- регулярки ---
CYR_RE = re.compile(r"^[А-Яа-яЁё]+$")
LAT_RE = re.compile(r"^[A-Za-z]+$")
DIGIT_RE = re.compile(r"^\d+$")
ONLY_LETTERS = re.compile(r"^[A-Za-z]+$")  # латиница без цифр/дефисов

# поддержка слитных процентов и объёмов
MERGED_PERCENT_RE = re.compile(r"^\d+[.,]?\d*%$")
MERGED_VOLUME_RE = re.compile(r"^\d+[.,]?\d*(л|ml|мл|г|гр|кг)$", re.IGNORECASE)

UNITS = {
    "л","литр","литра","литров",
    "ml","мл","миллилитр",
    "г","гр","грамм","грамма","граммов",
    "кг","килограмм",
    "%","процент","процентов"
}


# --------------------
# token utils
# --------------------
def clean_token(tok: str) -> str:
    """Приводим к нижнему регистру и убираем лишние символы."""
    return re.sub(r"[^a-zа-яё0-9-]+", "", tok.lower())


def is_digit(tok: str) -> bool:
    """Только цифры."""
    return DIGIT_RE.match(tok) is not None


def is_volume_token(tok: str) -> bool:
    """Токен выглядит как объем (слитный или число+единица)."""
    t = tok.lower()
    if MERGED_VOLUME_RE.match(t):
        return True
    return t in UNITS


def is_percent_token(tok: str) -> bool:
    """Токен выглядит как процент (слитный или %)."""
    t = tok.lower()
    if MERGED_PERCENT_RE.match(t):
        return True
    return t in {"%","процент","процентов"}


def normalize_tokens(tokens):
    """Нормализуем все токены (ё → е, lowercase)."""
    return [normalize_yo(t).lower() for t in tokens]


def mark_volume(tokens):
    """
    Ставим BIO-разметку для объёмов внутри токенов.
    Число + единица → B-VOLUME I-VOLUME
    'большой/средний/малый объем' → B-VOLUME I-VOLUME
    """
    labels = []
    i = 0
    while i < len(tokens):
        tok = tokens[i].lower()

        # число
        if re.match(r"^\d+[.,]?\d*$", tok):
            labels.append("B-VOLUME")
            if i+1 < len(tokens) and tokens[i+1].lower() in UNITS:
                labels.append("I-VOLUME")
                i += 2
                continue
            i += 1
            continue

        # словесный объем
        if tok in {"большой","средний","малый"} and i+1 < len(tokens) and tokens[i+1].lower() in {"объем","объём"}:
            labels.append("B-VOLUME")
            labels.append("I-VOLUME")
            i += 2
            continue

        if tok in {"объем","объём"}:
            labels.append("B-VOLUME")
            i += 1
            continue

        labels.append("O")
        i += 1

    return labels
