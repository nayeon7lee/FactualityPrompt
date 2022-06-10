import gzip
import os
import pickle
import re

import nltk
import numpy as np

from common.util.log_helper import LogHelper

# import torch
np.random.seed(55)


def vocab_map(vocab):
    voc_dict = {}
    for i, v in enumerate(vocab):
        voc_dict[v] = i
    # else:
    #     voc_dict['UNK'] = i
    return voc_dict


def tokenize(sequence):
    tokens = [token.replace("``", '').replace("''", '').replace('"', '') for token in nltk.word_tokenize(sequence) if
              token != " "]
    # return tokens
    return tokens


def clean_text(text):
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'[_"()|+&=*#$@\[\]/]', '', text)
    text = re.sub(r'\-', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = text.replace("...", " ")
    return text


def load_whole_glove(glove_file):
    logger = LogHelper.get_logger("load_whole_glove")
    is_gz = os.path.splitext(glove_file)[1] == '.gz'

    # Getting embedding dimension
    def _get_dim(_file):
        line = _file.readline()
        return len(line.strip().split(' ')) - 1

    if is_gz:
        with gzip.open(glove_file, 'rt') as file0:
            emb_dim = _get_dim(file0)
    else:
        with open(glove_file, 'r', encoding='utf-8') as file0:
            emb_dim = _get_dim(file0)

    # First row of embedding matrix is 0 for zero padding
    vocab = ['[PAD]']
    embed = [[0.0] * emb_dim]
    vocab.append('UNK')
    embed.append([1.0] * emb_dim)

    def _read_glove_file(_vocab, _embed, _file):
        for line in _file:
            items = line.replace('\r', '').replace('\n', '').split(' ')
            if len(items) < 10:
                logger.debug("exceptional line: {}".format(line))
                continue
            word = items[0]
            _vocab.append(word)
            vec = [float(i) for i in items[1:]]
            _embed.append(vec)
        return _vocab, _embed

    # Reading embedding matrix
    if is_gz:
        with gzip.open(glove_file, 'rt') as file:
            vocab, embed = _read_glove_file(vocab, embed, file)
    else:
        with open(glove_file, 'r', encoding='utf-8') as file:
            vocab, embed = _read_glove_file(vocab, embed, file)
    logger.info('Loaded GloVe!')
    return vocab, embed
# if __name__=="__main__":
#
#     text ="I don\'t think this is right..."
#     text =clean_text(text)
#     print(text)
