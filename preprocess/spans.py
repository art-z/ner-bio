from .utils import normalize_yo
from .tokens import clean_tokens, restore_brands, fix_bio_sequence, fix_numbers

def parse_spans(s: str):
    try:
        return [(int(a), int(b), str(c)) for (a, b, c) in eval(s)]
    except Exception:
        return []

def spans_to_bio(sample, spans, stopwords, brands):
    tokens = [normalize_yo(sample[s:e]) for s, e, _ in spans]
    labels = [lab for _, _, lab in spans]

    cleaned = clean_tokens(tokens, labels, stopwords)
    cleaned_labels = [lab for lab, _ in cleaned]
    restored = restore_brands(tokens, cleaned_labels, brands)
    bio = fix_bio_sequence(tokens, restored, stopwords)
    bio = fix_numbers(tokens, bio, brands)

    tokens = [normalize_yo(t) for t in tokens]
    return tokens, bio
