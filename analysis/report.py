import pandas as pd
from collections import Counter, defaultdict
import ast

from preprocess.utils import normalize_yo, is_cyrillic
from preprocess.stopwords import build_stopwords
from preprocess.brands import build_brands
from preprocess.utils import collect_adj_for_nouns
from augment.adj import extract_adjectives
from augment.noun_vol import extract_nouns


IN_FILE = "data/train.csv"
REPORT_FILE = "data/analysis_report.txt"

LIQUIDS = {"вода", "сок", "молоко"}
BULK    = {"хлеб", "масло", "мясо"}


def parse_spans(s: str):
    try:
        return [(int(a), int(b), str(c)) for (a, b, c) in ast.literal_eval(s)]
    except Exception:
        return []


def analyze():
    df = pd.read_csv(IN_FILE, sep=";")

    # --- статистика BIO меток ---
    tag_counts = Counter()
    for _, row in df.iterrows():
        spans = parse_spans(row["annotation"])
        for _, _, lab in spans:
            tag_counts[lab] += 1

    # --- бренды ---
    brands = build_brands(df)
    top_brands = brands.most_common(30)

    # --- существительные и прилагательные ---
    nouns = extract_nouns(df, max_nouns=50)
    adjs  = extract_adjectives(df, max_adj=50)

    # --- прилагательные для жидкостей и сыпучих ---
    brand_vocab = set(brands.keys())
    adj_liquids = collect_adj_for_nouns(df, LIQUIDS, brand_vocab)
    adj_bulk    = collect_adj_for_nouns(df, BULK, brand_vocab)

    # --- редкие/подозрительные ---
    short_cyr = []
    for _, row in df.iterrows():
        spans = parse_spans(row["annotation"])
        toks = [row["sample"][s:e] for s, e, _ in spans]
        labs = [lab for _, _, lab in spans]
        for tok, lab in zip(toks, labs):
            if is_cyrillic(tok) and len(tok) <= 2 and lab != "O":
                short_cyr.append((tok, lab, row["sample"]))

    # --- сохраняем ---
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("=== BIO Tag Counts ===\n")
        for tag, c in tag_counts.most_common():
            f.write(f"{tag}: {c}\n")

        f.write("\n=== Top Brands ===\n")
        for b, c in top_brands:
            f.write(f"{b}: {c}\n")

        f.write("\n=== Top Nouns (TYPE) ===\n")
        f.write(", ".join(nouns) + "\n")

        f.write("\n=== Top Adjectives (TYPE) ===\n")
        f.write(", ".join(adjs) + "\n")

        f.write("\n=== Adjectives for Liquids ===\n")
        for noun, adjs in adj_liquids.items():
            f.write(f"{noun}: {', '.join(adjs)}\n")

        f.write("\n=== Adjectives for Bulk ===\n")
        for noun, adjs in adj_bulk.items():
            f.write(f"{noun}: {', '.join(adjs)}\n")

        f.write("\n=== Suspicious short Cyrillic tokens (≤2) ===\n")
        for tok, lab, sample in short_cyr[:50]:
            f.write(f"{tok} → {lab} | {sample}\n")

    print(f"[✓] Report saved to {REPORT_FILE}")


if __name__ == "__main__":
    analyze()
