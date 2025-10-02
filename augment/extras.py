import random
import itertools

def augment_two_letter_cyr(max_aug=200, seed=42):
    random.seed(seed)
    letters = list("абвгдеёжзийклмнопрстуфхцчшщьыэюя")
    combos = ["".join(p) for p in itertools.product(letters, repeat=2)]
    random.shuffle(combos)
    combos = combos[:max_aug]
    aug_samples = [{"id": f"aug_cyr2_{i}", "search_query": c, "annotation": "O"} for i, c in enumerate(combos)]
    return aug_samples
