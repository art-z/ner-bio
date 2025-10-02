import random
from collections import Counter
import re

# только латиница
ONLY_LETTERS = re.compile(r"^[A-Za-z]+$")

def augment_split_brands(
    brands,
    min_first=2,
    max_first=3,
    min_len=6,
    max_aug=300,
    sample_ratio=0.3,
    seed=42,
):
    """
    Аугментация брендов: разрезаем латиницу на две части.
    Например: 'cocacola' → 'co coca'
    """
    random.seed(seed)
    aug_samples = []

    candidates = [
        b.lower() for b in brands
        if ONLY_LETTERS.match(b) and len(b) >= min_len
    ]

    k = int(len(candidates) * sample_ratio)
    if k > 0 and len(candidates) > 0:
        candidates = random.sample(candidates, k)

    for b in candidates:
        for cut in range(min_first, min(max_first + 1, len(b))):
            left, right = b[:cut], b[cut:]
            if len(right) < 2:
                continue
            aug_samples.append({
                "id": f"aug_brand_{b}_{cut}",
                "search_query": f"{left} {right}",
                "annotation": "B-BRAND I-BRAND"
            })

    if len(aug_samples) > max_aug:
        aug_samples = random.sample(aug_samples, max_aug)

    return aug_samples


def is_cyrillic(tok: str) -> bool:
    return bool(re.search(r"[А-Яа-яЁё]", tok))


def augment_split_brands_cyr(
    brands,
    min_first=2,
    max_first=3,
    min_len=6,
    max_aug=200,
    sample_ratio=0.3,
    seed=42,
):
    """
    Аугментация брендов: разрезаем кириллические слова (без пробелов).
    Например: 'пепсикола' → 'пе пси'
    """
    random.seed(seed)
    aug_samples = []

    candidates = [
        b.lower() for b in brands
        if is_cyrillic(b) and len(b) >= min_len and " " not in b
    ]

    k = int(len(candidates) * sample_ratio)
    if k > 0 and len(candidates) > 0:
        candidates = random.sample(candidates, k)

    for b in candidates:
        for cut in range(min_first, min(max_first + 1, len(b))):
            left, right = b[:cut], b[cut:]
            if len(right) < 2:
                continue
            aug_samples.append({
                "id": f"aug_cyr_{b}_{cut}",
                "search_query": f"{left} {right}",
                "annotation": "B-BRAND I-BRAND"
            })

    if len(aug_samples) > max_aug:
        aug_samples = random.sample(aug_samples, max_aug)

    return aug_samples
