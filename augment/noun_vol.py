import random

def extract_nouns(df, max_nouns=100):
    from collections import Counter
    import pymorphy2
    morph = pymorphy2.MorphAnalyzer()
    nouns = Counter()
    for _, row in df.iterrows():
        sample = str(row["sample"]).lower()
        spans = eval(row["annotation"])
        toks = [sample[s:e].strip() for s, e, _ in spans]
        labs = [lab for _, _, lab in spans]
        for tok, lab in zip(toks, labs):
            if lab.endswith("TYPE"):
                for p in morph.parse(tok):
                    if p.tag.POS == "NOUN":
                        nouns[tok] += 1; break
    return [n for n, _ in nouns.most_common(max_nouns)]

def augment_noun_with_volumes(nouns, adjectives, max_aug=300, percent=0.1, seed=42):
    random.seed(seed)
    aug_samples = []
    VOLUMES = ["1 л","1 литр","2 литра","0.5 л","500 мл","330 мл",
               "1000 мл","100 г","200 г","500 грамм","1 кг","2 кг",
               "большой объем","средний объем","малый объем"]
    k = int(len(nouns) * percent)
    nouns = random.sample(nouns, min(k, len(nouns)))
    idx = 0
    for noun in nouns:
        adj = random.choice(adjectives) if random.random() < 0.4 else None
        vol = random.choice(VOLUMES)
        toks = [noun] + ([adj] if adj else []) + vol.split()
        labs = ["B-TYPE"] + (["I-TYPE"] if adj else []) + ["B-VOLUME"] + ["I-VOLUME"]*(len(vol.split())-1)
        aug_samples.append({"id": f"aug_vol_{idx}", "search_query": " ".join(toks), "annotation": " ".join(labs)})
        idx += 1
        if idx >= max_aug: break
    return aug_samples
