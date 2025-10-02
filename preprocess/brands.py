from collections import Counter, defaultdict
from .utils import normalize_yo

def build_brands(df):
    phrases, singles = Counter(), Counter()
    for _, row in df.iterrows():
        sample = normalize_yo(str(row["sample"]))
        spans = eval(row["annotation"])
        toks = [normalize_yo(sample[s:e].strip().lower()) for s, e, _ in spans]
        labs = [lab for _, _, lab in spans]
        i = 0
        while i < len(toks):
            if "BRAND" in labs[i]:
                run, j = [], i
                while j < len(toks) and "BRAND" in labs[j]:
                    run.append(toks[j]); j += 1
                if len(run) == 1:
                    w = run[0]; singles[w] += 1
                    if "-" in w: singles[w.replace("-", " ")] += 1
                else:
                    phrase = " ".join(run); phrases[phrase] += 1
                i = j
            else:
                i += 1
    brands = Counter(); brands.update(singles); brands.update(phrases)
    return brands

def filter_brands_by_majority(df, brands):
    stats, head_b_counts = defaultdict(Counter), Counter()
    for _, row in df.iterrows():
        sample = normalize_yo(str(row["sample"]))
        spans = eval(row["annotation"])
        toks = [normalize_yo(sample[s:e].strip().lower()) for s, e, _ in spans]
        labs = [lab for _, _, lab in spans]
        for tok, lab in zip(toks, labs): stats[tok][lab] += 1
        i = 0
        while i < len(toks):
            if labs[i] == "B-BRAND":
                j = i+1
                while j < len(toks) and labs[j] == "I-BRAND": j += 1
                if j-i == 1: head_b_counts[toks[i]] += 1
                i = j
            else: i += 1
    filtered = Counter()
    for key, cnt in brands.items():
        if " " in key: filtered[key] = cnt
        else:
            b_head = head_b_counts[key]
            b_any  = stats[key]["B-BRAND"]
            i_any  = stats[key]["I-BRAND"]
            other  = sum(stats[key].values()) - (b_any+i_any)
            if b_head >= 1 and b_any >= i_any and b_any >= other:
                filtered[key] = cnt
    return filtered
