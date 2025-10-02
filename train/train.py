
import os, re, random
import numpy as np
import pandas as pd

from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    DataCollatorForTokenClassification, Trainer, TrainingArguments
)
from seqeval.metrics import f1_score, precision_score, recall_score
import torch

MODEL_NAME = os.environ.get("MODEL_NAME", "DeepPavlov/rubert-base-cased")
#MODEL_NAME = os.path.join(os.path.dirname(__file__), "rubert-base-cased")

SEED = int(os.environ.get("SEED", 42))

labels = ["O","B-TYPE","I-TYPE","B-BRAND","I-BRAND","B-VOLUME","I-VOLUME","B-PERCENT","I-PERCENT"]
id2label = {i:l for i,l in enumerate(labels)}
label2id = {l:i for i,l in enumerate(labels)}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def check_and_split(row):

    raw_toks = row["search_query"].split()
    toks = [tok.strip("\"'«»“”„") for tok in raw_toks]
    toks = [t for t in toks if t]

    tags = row["annotation"].split()

    # --- Чистим метки ---
    clean_tags = []
    for t in tags:
        if t == "0":
            clean_tags.append("O")
        elif t in label2id:
            clean_tags.append(t)
        else:
            raise ValueError(f"Неизвестная метка: {t} (id={row['id']})")

    if len(toks) != len(clean_tags):
        raise ValueError(
            f"Длины не совпали (id={row['id']}): {len(toks)} токенов vs {len(clean_tags)} тегов\n"
            f"query={row['search_query']} toks={toks} tags={clean_tags}"
        )

    return {"tokens": toks, "ner_tags": [label2id[t] for t in clean_tags]}

def align_labels(ex, tok):
    tokenized = tok(ex["tokens"], is_split_into_words=True, truncation=True, max_length=256)
    word_ids = tokenized.word_ids()
    aligned = []
    prev = None
    for wi in word_ids:
        if wi is None:
            aligned.append(-100)
        else:
            if wi != prev:
                aligned.append(ex["ner_tags"][wi])
            else:
                aligned.append(-100)
            prev = wi
    tokenized["labels"] = aligned
    return tokenized

def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=-1)
    true = eval_pred.label_ids

    pred_tags, true_tags = [], []
    for p_seq, t_seq in zip(preds, true):
        p_line, t_line = [], []
        for p_i, t_i in zip(p_seq, t_seq):
            if t_i == -100:
                continue
            p_line.append(id2label[p_i])
            t_line.append(id2label[t_i])
        pred_tags.append(p_line)
        true_tags.append(t_line)

    return {
        "precision": precision_score(true_tags, pred_tags),
        "recall": recall_score(true_tags, pred_tags),
        "f1_macro": f1_score(true_tags, pred_tags, average="macro")
    }

def main():
    set_seed(SEED)

    df = pd.read_csv("data/train_bio_final.csv")

    base = Dataset.from_pandas(df)
    base = base.map(check_and_split, remove_columns=df.columns.tolist())

    # Загружаем токенизатор и модель с safetensors
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=True)
    tokenized = base.map(lambda ex: align_labels(ex, tok), batched=False)

    split = tokenized.train_test_split(test_size=0.1, seed=SEED)
    train_ds, eval_ds = split["train"], split["test"]

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        trust_remote_code=True,
        use_safetensors=True
    )

    collator = DataCollatorForTokenClassification(tok)

    args = TrainingArguments(
        output_dir="model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        report_to="none",
        fp16=True,
        logging_strategy="steps",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("model")
    tok.save_pretrained("model")

if __name__ == "__main__":
    main()
