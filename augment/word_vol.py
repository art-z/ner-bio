import random

def augment_word_volumes(nouns, adjectives, max_aug=300, seed=42):
    random.seed(seed)
    VOLUME_WORDS = ["большой объем", "средний объем", "малый объем"]
    aug_samples, idx = [], 0
    for noun in nouns:
        for adj in adjectives[:3]:
            for vw in VOLUME_WORDS:
                toks = [noun, adj] + vw.split()
                labs = ["B-TYPE","I-TYPE","B-VOLUME","I-VOLUME"]
                aug_samples.append({
                    "id": f"aug_wordvol_{idx}",
                    "search_query": " ".join(toks),
                    "annotation": " ".join(labs)
                }); idx += 1
                if idx >= max_aug: return aug_samples
    return aug_samples

def augment_adj_only_volumes(adjectives, max_aug=200, seed=42):
    random.seed(seed)
    NUM_VOLUMES = ["500 мл","330 мл","1000 мл","1 л","2 литра","1 кг","500 грамм"]
    WORD_VOLUMES = ["объем","большой объем","средний объем","малый объем"]
    aug_samples, idx = [], 0
    for adj in adjectives[:20]:
        vol = random.choice(NUM_VOLUMES)
        toks = [adj] + vol.split()
        labs = ["B-TYPE","B-VOLUME"] + ["I-VOLUME"]*(len(vol.split())-1)
        aug_samples.append({
            "id": f"aug_adjvol_num_{idx}",
            "search_query": " ".join(toks),
            "annotation": " ".join(labs)
        }); idx += 1
        for word in WORD_VOLUMES:
            toks = [adj] + word.split()
            labs = ["B-TYPE","B-VOLUME"] + ["I-VOLUME"]*(len(toks)-2 if len(toks)>2 else 0)
            aug_samples.append({
                "id": f"aug_adjvol_word_{idx}",
                "search_query": " ".join(toks),
                "annotation": " ".join(labs)
            }); idx += 1
        if idx >= max_aug: break
    return aug_samples
