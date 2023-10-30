import json
import torch as th
from nltk.tokenize import word_tokenize


def text_to_tokens(text, vocab):
    seq = []
    words = word_tokenize(text)
    for word in words:
        if word not in vocab:
            seq.append(vocab['<UNK>'])
        else:
            seq.append(vocab[word])

    return th.tensor(seq, dtype=th.long)


def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
    return vocab 