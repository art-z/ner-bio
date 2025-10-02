import random
import itertools
import re

def augment_volume_percent(volumes=None, percents=None, max_aug=200, seed=42):
    random.seed(seed)
    aug_samples = []
    if volumes is None:
        volumes = ["0.5 л", "1 л", "1.5 л", "2 л", "330 мл", "500 мл",
                   "750 мл", "1000 мл", "200 г", "500 г", "1 кг"]
    if percents is None:
        percents = ["1%", "1.5%", "2%", "2.5%", "3%", "3.2%", "6%", "10%", "15%", "25%"]

    for i, v in enumerate(volumes):
        toks = v.split()
        labs = ["B-VOLUME"] + ["I-VOLUME"]*(len(toks)-1)
        aug_samples.append({"id": f"aug_volume_{i}", "search_query": v, "annotation": " ".join(labs)})
    for i, p in enumerate(percents):
        toks = p.split()
        labs = ["B-PERCENT"] + ["I-PERCENT"]*(len(toks)-1)
        aug_samples.append({"id": f"aug_percent_{i}", "search_query": p, "annotation": " ".join(labs)})
    return aug_samples[:max_aug]

def augment_pure_volumes(seed=42):
    random.seed(seed)
    phrases = [
        "большой объем", "средний объем", "малый объем",
        "средний объём", "большой объём", "малый объём",
        "пять литров", "1 литр", "500 грамм", "2 кг"
    ]
    aug_samples = []
    for i, ph in enumerate(phrases):
        toks = ph.split()
        labs = ["B-VOLUME"] + ["I-VOLUME"]*(len(toks)-1)
        aug_samples.append({"id": f"aug_purevol_{i}", "search_query": ph, "annotation": " ".join(labs)})
    return aug_samples
