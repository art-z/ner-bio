import pandas as pd

# --- наши модули ---
from preprocess.stopwords import build_stopwords
from preprocess.brands import build_brands, filter_brands_by_majority
from preprocess.bio_fix import spans_to_bio
from augment.brands import augment_split_brands, augment_split_brands_cyr
from augment.volumes import augment_volume_percent, augment_pure_volumes
from augment.adj import extract_adjectives, augment_broken_adj_types_from_train
from augment.noun_vol import extract_nouns, augment_noun_with_volumes
from augment.word_vol import augment_word_volumes, augment_adj_only_volumes
from augment.extras import augment_two_letter_cyr
from preprocess.utils import collect_adj_for_nouns


IN_FILE = "data/train.csv"
OUT_FILE = "data/train_bio_final.csv"
STOPWORDS_FILE = "data/stopwords.txt"
BRANDS_FILE = "data/brands.txt"

LIQUIDS = {"вода", "сок", "молоко"}
BULK    = {"хлеб", "масло", "мясо"}


def main():
    df = pd.read_csv(IN_FILE, sep=";")

    # --- Step 1: стоп-слова ---
    stopwords = build_stopwords(df)
    with open(STOPWORDS_FILE, "w", encoding="utf-8") as f:
        for w in sorted(stopwords):
            f.write(w + "\n")
    print(f"[✓] Stopwords saved to {STOPWORDS_FILE}")

    # --- Step 2: бренды ---
    brands = build_brands(df)
    brands = filter_brands_by_majority(df, brands)
    with open(BRANDS_FILE, "w", encoding="utf-8") as f:
        for b, c in brands.most_common():
            f.write(f"{b}\t{c}\n")
    print(f"[✓] Brands saved to {BRANDS_FILE}")

    # --- Step 3: базовые BIO ---
    out_rows = []
    for i, row in df.iterrows():
        sample = str(row["sample"])
        spans = eval(row["annotation"])
        tokens, bio = spans_to_bio(sample, spans, stopwords, brands)
        out_rows.append({
            "id": i,
            "search_query": " ".join(tokens),
            "annotation": " ".join(bio)
        })
    out_df = pd.DataFrame(out_rows)

    # --- Step 4: аугментации ---
    aug_df = pd.DataFrame(
        augment_split_brands(list(brands.keys()), max_aug=100) +
        augment_split_brands_cyr(list(brands.keys()), max_aug=100)
    )

    aug_df2 = pd.DataFrame(augment_volume_percent())
    aug_df3 = pd.DataFrame(augment_two_letter_cyr(max_aug=200))

    adjectives = extract_adjectives(df, max_adj=50)
    nouns = extract_nouns(df, max_nouns=100)

    aug_df4 = pd.DataFrame(
        augment_broken_adj_types_from_train(adjectives, nouns, df_size=len(df), percent=0.1)
    )
    aug_df5 = pd.DataFrame(
        augment_noun_with_volumes(nouns, adjectives, max_aug=200, percent=0.2)
    )

    # --- Step 5: прилагательные для жидкостей/сыпучих ---
    brand_vocab = set(brands.keys())
    adj_liquids = collect_adj_for_nouns(df, LIQUIDS, brand_vocab=brand_vocab)
    adj_bulk    = collect_adj_for_nouns(df, BULK, brand_vocab=brand_vocab)

    with open("data/liquid_adj.txt","w",encoding="utf-8") as f:
        for noun, adjs in adj_liquids.items():
            f.write(f"{noun}: {', '.join(adjs)}\n")

    with open("data/bulk_adj.txt","w",encoding="utf-8") as f:
        for noun, adjs in adj_bulk.items():
            f.write(f"{noun}: {', '.join(adjs)}\n")

    print("[✓] Liquid adjectives saved to data/liquid_adj.txt")
    print("[✓] Bulk adjectives saved to data/bulk_adj.txt")

    # --- Step 6: word-volumes ---
    aug_df6 = pd.DataFrame(augment_word_volumes(nouns, adjectives, max_aug=700))
    aug_df7 = pd.DataFrame(augment_adj_only_volumes(adjectives, max_aug=700))
    aug_df8 = pd.DataFrame(augment_pure_volumes())

    # --- Step 7: объединение ---
    final_df = pd.concat([
        out_df, aug_df, aug_df2, aug_df3,
        aug_df4, aug_df5, aug_df6, aug_df7, aug_df8
    ], ignore_index=True)

    final_df.to_csv(OUT_FILE, index=False)
    print(f"[✓] BIO dataset (with augmentation) saved to {OUT_FILE}")

    print("\n[Примеры 'объем']:")
    print(final_df[final_df["search_query"].str.contains("объем")].sample(10))


if __name__ == "__main__":
    main()
