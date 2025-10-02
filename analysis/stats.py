import pandas as pd
from collections import Counter
import ast
import json

from augment.adj import extract_adjectives
from augment.noun_vol import extract_nouns


IN_FILE = "data/train.csv"
STATS_JSON = "data/stats.json"
STATS_CSV  = "data/stats.csv"


def parse_spans(s: str):
    try:
        return [(int(a), int(b), str(c)) for (a, b, c) in ast.literal_eval(s)]
    except Exception:
        return []


def collect_stats():
    df = pd.read_csv(IN_FILE, sep=";")

    # --- Общее ---
    total_samples = len(df)

    # --- BIO теги ---
    tag_counts = Counter()
    for _, row in df.iterrows():
        for _, _, lab in parse_spans(row["annotation"]):
            tag_counts[lab] += 1

    # --- Существительные и прилагательные ---
    nouns = extract_nouns(df, max_nouns=100)
    adjs  = extract_adjectives(df, max_adj=100)

    # --- Биграммы "adj + noun" ---
    bigrams = Counter()
    for _, row in df.iterrows():
        toks = row["sample"].split()
        for i in range(len(toks) - 1):
            pair = (toks[i].lower(), toks[i+1].lower())
            bigrams[pair] += 1
    top_bigrams = bigrams.most_common(50)

    # --- Готовим результат ---
    stats = {
        "total_samples": total_samples,
        "tag_counts": dict(tag_counts),
        "top_nouns": nouns,
        "top_adjectives": adjs,
        "top_bigrams": [{"pair": " ".join(p), "count": c} for p, c in top_bigrams]
    }

    # --- Сохраняем ---
    with open(STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # CSV для Excel
    rows = []
    for tag, c in tag_counts.items():
        rows.append({"type": "tag", "name": tag, "count": c})
    for n in nouns:
        rows.append({"type": "noun", "name": n, "count": None})
    for adj in adjs:
        rows.append({"type": "adj", "name": adj, "count": None})
    for bg, c in top_bigrams:
        rows.append({"type": "bigram", "name": " ".join(bg), "count": c})

    pd.DataFrame(rows).to_csv(STATS_CSV, index=False)

    print(f"[✓] Stats saved to {STATS_JSON} and {STATS_CSV}")


if __name__ == "__main__":
    collect_stats()
