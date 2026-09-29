import re
from collections import Counter

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\b\d{7,}\b", " PHONE ", text)
    text = re.sub(r"[^a-z0-9\s!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str):
    # Simple educational tokenizer; Regex is used before tokenization.
    return re.findall(r"[a-z0-9]+|[!?]", clean_text(text))

def build_vocab(texts, min_freq=1):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, freq in sorted(counter.items()):
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def encode(text, vocab):
    return [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokenize(text)]

def pad_sequence(ids, max_len, pad_idx=0):
    ids = ids[:max_len]
    return ids + [pad_idx] * (max_len - len(ids))