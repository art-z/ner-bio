import pymorphy2
from .utils import normalize_yo

morph = pymorphy2.MorphAnalyzer()

def is_function_word(tok: str) -> bool:
    if not tok or len(tok) > 3:
        return False
    for p in morph.parse(tok):
        if p.tag.POS in {"PREP", "CONJ", "PRCL"}:
            return True
    return False

def build_stopwords(df):
    stopwords = set()
    for sample in df["sample"]:
        for tok in str(sample).split():
            if is_function_word(tok):
                stopwords.add(normalize_yo(tok.lower()))
    return stopwords
