import re
from .utils import is_function_word, is_cyrillic, is_latin, token_has_digit, normalize_yo

# --------------------
# clean tokens
# --------------------
def clean_tokens(tokens, labels, stopwords):
    fixed = []
    is_single_word_query = (len(tokens) == 1)

    for i, (tok, lab) in enumerate(zip(tokens, labels)):
        t = tok.lower()

        # стоп-слово → O
        if t in stopwords:
            fixed.append(("O", tok))
            continue

        # число без единиц → O
        if t.isdigit() and not lab.endswith(("VOLUME", "PERCENT")):
            fixed.append(("O", tok))
            continue

        # одиночная кириллица длиной 2 буквы → O
        if is_single_word_query and re.fullmatch(r"[а-яе]{2}", t, flags=re.IGNORECASE):
            fixed.append(("O", tok))
            continue

        # короткая кириллица 2–3 буквы (но не союз/предлог) → TYPE
        if (
            is_single_word_query and lab == "O" and 2 <= len(t) <= 3
            and re.fullmatch(r"[а-яе]+", t, flags=re.IGNORECASE)
            and not is_function_word(tok) and t not in stopwords
        ):
            fixed.append(("B-TYPE", tok))
            continue

        # валидные BIO
        if lab.endswith(("TYPE", "BRAND", "VOLUME", "PERCENT")) or lab == "O":
            fixed.append((lab, tok))
        else:
            fixed.append(("O", tok))

    return fixed


# --------------------
# restore brands
# --------------------
def restore_brands(tokens, labels, brands):
    fixed = labels[:]
    toks = [tok.lower() for tok in tokens]
    n = len(toks)

    # --- многословные бренды ---
    brand_phrases = [b.split() for b in brands if " " in b]
    for bt in brand_phrases:
        L = len(bt)
        if L == 0 or L > n:
            continue
        for i in range(n - L + 1):
            if toks[i:i+L] == bt:
                fixed[i] = "B-BRAND"
                for j in range(1, L):
                    fixed[i+j] = "I-BRAND"

    # --- одиночные бренды ---
    single_brands = {b for b in brands if " " not in b}
    for i, tok in enumerate(toks):
        if tok in single_brands:
            if fixed[i].startswith("I-BRAND"):
                continue
            if i > 0 and fixed[i-1].endswith("TYPE"):  # "каша ночь"
                continue
            fixed[i] = "B-BRAND"

    return fixed


# --------------------
#  BIO sequence
# --------------------
def fix_bio_sequence(tokens, labels, stopwords=None):
    fixed = labels[:]

    for i in range(len(fixed) - 1):
        # B-BRAND B-BRAND → I-BRAND
        if fixed[i] == "B-BRAND" and fixed[i+1] == "B-BRAND":
            fixed[i+1] = "I-BRAND"

        # B-TYPE B-TYPE → I-TYPE
        if fixed[i] == "B-TYPE" and fixed[i+1] == "B-TYPE":
            fixed[i+1] = "I-TYPE"

        # B-TYPE O I-TYPE → B-TYPE I-TYPE I-TYPE
        if fixed[i] == "B-TYPE" and fixed[i+1] == "O":
            if i+2 < len(fixed) and fixed[i+2] == "I-TYPE":
                fixed[i+1] = "I-TYPE"

        # BRAND смешанный алфавит → сброс
        if fixed[i] == "B-BRAND" and fixed[i+1] == "I-BRAND":
            if (is_latin(tokens[i]) and is_cyrillic(tokens[i+1])) or \
               (is_cyrillic(tokens[i]) and is_latin(tokens[i+1])):
                fixed[i+1] = "B-TYPE"

    # мусорные бренды (односимвольные, цифры)
    for i, (tok, lab) in enumerate(zip(tokens, fixed)):
        if lab == "B-BRAND" and (len(tok) <= 1 or tok.isdigit()):
            fixed[i] = "O"

    # ограничиваем I-TYPE после B-TYPE
    type_open = False
    for i in range(len(fixed)):
        if fixed[i] == "B-TYPE":
            type_open = True
        elif fixed[i] == "I-TYPE":
            if not type_open:
                fixed[i] = "O"
            else:
                type_open = False
        else:
            type_open = False

    return fixed


# --------------------
#  numbers / volumes
# --------------------
def fix_numbers(tokens, labels, brands):
    fixed = labels[:]
    n = len(tokens)

    for i, tok in enumerate(tokens):
        tok_l = tok.lower()

        # слитные проценты
        if re.match(r"^\d+[.,]?\d*%$", tok_l):
            fixed[i] = "B-PERCENT"
            continue

        # слитные объемы
        if re.match(r"^\d+[.,]?\d*(л|ml|мл|г|гр|кг)$", tok_l):
            fixed[i] = "B-VOLUME"
            continue

        # числа + единицы
        if tok.isdigit():
            if i+1 < n and tokens[i+1].lower() in {"л","литр","литра","литров","мл","г","гр","грамм","кг"}:
                fixed[i] = "B-VOLUME"
                fixed[i+1] = "I-VOLUME"
                continue
            if i+1 < n and tokens[i+1].lower() in {"%","процент","процентов"}:
                fixed[i] = "B-PERCENT"
                fixed[i+1] = "I-PERCENT"
                continue

    return fixed


# --------------------
#  spans → BIO
# --------------------
def spans_to_bio(sample, spans, stopwords, brands):
    tokens = [normalize_yo(sample[s:e]) for s, e, _ in spans]  # ⚡ нормализуем ё→е
    labels = [lab for _, _, lab in spans]

    cleaned = clean_tokens(tokens, labels, stopwords)
    cleaned_labels = [lab for lab, _ in cleaned]

    restored = restore_brands(tokens, cleaned_labels, brands)
    bio = fix_bio_sequence(tokens, restored, stopwords)
    bio = fix_numbers(tokens, bio, brands)

    return tokens, bio
