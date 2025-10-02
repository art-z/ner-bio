import random
import pymorphy2
from collections import Counter

morph = pymorphy2.MorphAnalyzer()

def extract_adjectives(df, max_adj=50):
    adjectives = Counter()
    for _, row in df.iterrows():
        sample = str(row["sample"]).lower()
        spans = eval(row["annotation"])
        toks = [sample[s:e].strip() for s, e, _ in spans]
        labs = [lab for _, _, lab in spans]
        for tok, lab in zip(toks, labs):
            if lab.endswith("TYPE"):
                for p in morph.parse(tok):
                    if p.tag.POS == "ADJF":
                        adjectives[tok] += 1; break
    return [adj for adj, _ in adjectives.most_common(max_adj)]

def augment_broken_adj_types_from_train(adjectives, nouns, df_size, percent=0.1, seed=42):
    random.seed(seed)
    aug_samples = []
    max_aug = int(df_size * percent)
    pairs = [(adj, noun) for adj in adjectives for noun in nouns]
    random.shuffle(pairs)
    pairs = pairs[:max_aug]
    idx = 0
    for adj, noun in pairs:
        if len(adj) > 2:
            toks = [adj[:2], adj[2:], noun]
            labs = ["B-TYPE", "I-TYPE", "I-TYPE"]
            aug_samples.append({
                "id": f"aug_brokenadj_{idx}",
                "search_query": " ".join(toks),
                "annotation": " ".join(labs)
            }); idx += 1
    return aug_samples
