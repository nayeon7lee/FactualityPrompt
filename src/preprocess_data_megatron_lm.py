'''
    Adapted from https://github.com/NVIDIA/Megatron-LM/blob/main/tools/preprocess_data.py by NVIDIA CORPORATION.
'''
# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Processing data for pretraining."""

import argparse
import json
import multiprocessing
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time

import random
import torch
try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False

import string
import numpy as np
import spacy
# python -m spacy download en_core_web_sm
spacy.load('en_core_web_sm')
nlp = spacy.load("en_core_web_sm")

from megatron.tokenizer import build_tokenizer
from megatron.data import indexed_dataset

from nltk.tokenize import sent_tokenize


# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text

class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)
        if self.args.split_sentences:
            if not nltk_available:
                print("NLTK is not available to split sentences.")
                exit()
            splitter = nltk.load("tokenizers/punkt/english.pickle")
            if self.args.keep_newlines:
                # this prevents punkt from eating newlines after sentences
                Encoder.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text = splitter._params,
                    lang_vars = CustomLanguageVars())
            else:
                Encoder.splitter = splitter

        else:
            Encoder.splitter = IdentitySplitter()

    def encode(self, json_line):
        data = json.loads(json_line)
        ids = {}
        for key in self.args.json_keys:
            # # original
            # text = data[key]
            # doc_ids = [] # list of tokenized_ids from whole doc
            # for sentence in Encoder.splitter.tokenize(text):
            #     sentence_ids = Encoder.tokenizer.tokenize(sentence)
            #     if len(sentence_ids) > 0:
            #         doc_ids.append(sentence_ids)
            # if len(doc_ids) > 0 and self.args.append_eod:
            #     doc_ids[-1].append(Encoder.tokenizer.eod)
            # ids[key] = doc_ids

            
            text = data['text']
            doc_ids = [] # list of tokenized_ids from whole doc
            doc_ne_mask_ids = [] # list of ne_mask for the whole doc


            for sentence in Encoder.splitter.tokenize(text):
                # sentence_ids = Encoder.tokenizer.tokenize(sentence)

                if self.args.mask_type == 'v1_all_NE':
                    # unmask all NE. rest are masked
                    ne_mask_ids, sentence_ids = make_ne_mask(sentence, Encoder.tokenizer)
                
                elif self.args.mask_type == 'v2_all_after_ROOT':
                    # unmask all after ROOT. substring before ROOT are masked
                    ne_mask_ids, sentence_ids = make_second_half_mask(sentence, Encoder.tokenizer, split_type='root')

                elif self.args.mask_type == 'v3_all_after_half':
                    ne_mask_ids, sentence_ids = make_second_half_mask(sentence, Encoder.tokenizer, split_type='half')

                elif self.args.mask_type == 'v4_NE_after_ROOT':
                    ne_mask_ids, sentence_ids = make_NE_after_root_mask(sentence, Encoder.tokenizer)
                
                elif self.args.mask_type == 'v5_RANDOM_Mask':
                    ne_mask_ids, sentence_ids = make_second_half_mask(sentence, Encoder.tokenizer, split_type='random')
                
                else:
                    print("Wrong mask type provided! Check setting!!")
                    exit(0)

                    
                if len(sentence_ids) > 0:
                    doc_ids.append(sentence_ids)
                    doc_ne_mask_ids.append(ne_mask_ids)

            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids[-1].append(Encoder.tokenizer.eod)
                doc_ne_mask_ids[-1].append(0) # mask the last eod token as well

            ids['text'] = doc_ids
            ids['ne_mask'] = doc_ne_mask_ids

        # print(len(ids['text'][0]),len(ids['ne_mask'][0]))
        assert len(ids['text'][0]) == len(ids['ne_mask'][0])

        return ids, len(json_line)

def make_second_half_mask(raw_text, tokenizer, split_type='root'):
    
    mask_ids = [] 
    sentence_ids = []

    for sent in sent_tokenize(raw_text):

        doc = nlp(sent)
        all_tokens = [tok.text for tok in doc]
        
        if split_type == 'root':
            split_idx = [ token.dep_ for token in doc].index('ROOT')

        elif split_type == 'half':
            split_idx = len(all_tokens) // 2
        
        elif split_type == 'random':
            split_idx = int(len(all_tokens) * random.uniform(0.25, 0.75))

        first_half = " " + " ".join(all_tokens[:split_idx+1])
        second_half = " " + " ".join(all_tokens[split_idx+1:])

        # mask first_half
        first_half_bpe = tokenizer.tokenize(first_half)
        sentence_ids.extend(first_half_bpe)
        mask_ids.extend([0] * len(first_half_bpe))

        # UNmask second_half
        second_half_bpe = tokenizer.tokenize(second_half)
        sentence_ids.extend(second_half_bpe)
        mask_ids.extend([1] * len(second_half_bpe))

    return mask_ids, sentence_ids
    

def make_ne_mask(raw_text, tokenizer):

    NE_mask_ids = [] 

    prev_end_idx = -1

    doc = nlp(raw_text)
    all_tokens = [tok.text for tok in doc]
    # print("all_tokens", all_tokens)
    nes_tokens = [ent for ent in doc.ents]
    # print("nes_tokens", nes_tokens, "\n")

    sentence_ids = []
    for ne in nes_tokens:
        # print(ne, ne.start, ne.end)
        cur_start_idx = ne.start
        cur_end_idx = ne.end

        preceding_non_ne = " ".join(all_tokens[prev_end_idx: cur_start_idx])

        if len(preceding_non_ne) == 0: # means cur NE is the beginning of sent.
            cur_ne = ne.text
        else:
            if preceding_non_ne[0] not in string.punctuation:
                preceding_non_ne = " " + preceding_non_ne
            cur_ne = " " + ne.text

        # print("preceding_non_ne", preceding_non_ne)
        # print("cur_ne", cur_ne, "\n")

        # mask non-nes between prev ne and cur-ne
        preceding_non_ne_bpe = tokenizer.tokenize(preceding_non_ne)
        sentence_ids.extend(preceding_non_ne_bpe)
        NE_mask_ids.extend([0] * len(preceding_non_ne_bpe))

        # unmask cur-ne 
        cur_ne_bqe = tokenizer.tokenize(cur_ne)
        sentence_ids.extend(cur_ne_bqe)
        NE_mask_ids.extend([1] * len(cur_ne_bqe))

        # update prev_end_idx 
        prev_end_idx = cur_end_idx
    
    last_non_ne = " ".join(all_tokens[prev_end_idx: len(all_tokens)])
    last_non_ne_bpe = tokenizer.tokenize(last_non_ne)
    sentence_ids.extend(last_non_ne_bpe)
    NE_mask_ids.extend([0] * len(last_non_ne_bpe))

    return NE_mask_ids, sentence_ids


def make_NE_after_root_mask(raw_text, tokenizer):

    NE_mask_ids = [] 

    prev_end_idx = -1

    doc = nlp(raw_text)

    root_idx = [ token.dep_ for token in doc].index('ROOT')
    # root defines the start of our "second half"

    all_tokens = [tok.text for tok in doc]
    # print("all_tokens", all_tokens)
    nes_tokens = [ent for ent in doc.ents]
    # print("nes_tokens", nes_tokens, "\n")

    # filter ne_tokens to only contain those AFTER root_idx
    nes_tokens = [ne for ne in nes_tokens if ne.start > root_idx ]


    sentence_ids = []
    for ne in nes_tokens:
        # print(ne, ne.start, ne.end)
        cur_start_idx = ne.start
        cur_end_idx = ne.end

        preceding_non_ne = " ".join(all_tokens[prev_end_idx: cur_start_idx])

        if len(preceding_non_ne) == 0: # means cur NE is the beginning of sent.
            cur_ne = ne.text
        else:
            if preceding_non_ne[0] not in string.punctuation:
                preceding_non_ne = " " + preceding_non_ne
            cur_ne = " " + ne.text

        # print("preceding_non_ne", preceding_non_ne)
        # print("cur_ne", cur_ne, "\n")

        # mask non-nes between prev ne and cur-ne
        preceding_non_ne_bpe = tokenizer.tokenize(preceding_non_ne)
        sentence_ids.extend(preceding_non_ne_bpe)
        NE_mask_ids.extend([0] * len(preceding_non_ne_bpe))

        # unmask cur-ne 
        cur_ne_bqe = tokenizer.tokenize(cur_ne)
        sentence_ids.extend(cur_ne_bqe)
        NE_mask_ids.extend([1] * len(cur_ne_bqe))

        # update prev_end_idx 
        prev_end_idx = cur_end_idx
    
    last_non_ne = " ".join(all_tokens[prev_end_idx: len(all_tokens)])
    last_non_ne_bpe = tokenizer.tokenize(last_non_ne)
    sentence_ids.extend(last_non_ne_bpe)
    NE_mask_ids.extend([0] * len(last_non_ne_bpe))

    return NE_mask_ids, sentence_ids


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--json-keys', nargs='+', default=['text', 'ne_mask'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')

    group.add_argument('--mask_type', type=str, help='Masking logic choice: [v1_all_NE, v2_all_after_ROOT, v3_all_after_half, v4_ne_latter]')
    
    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, required=True,
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')


    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes to launch')
    group.add_argument('--log-interval', type=int, default=100,
                       help='Interval between progress updates')

    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert'):
        if not args.split_sentences:
            print("Bert tokenizer detected, are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args

def main():
    args = get_args()
    startup_start = time.time()

    print("Opening", args.input)
    fin = open(args.input, 'r', encoding='utf-8')

    if nltk_available and args.split_sentences:
        nltk.download("punkt", quiet=True)

    encoder = Encoder(args)
    tokenizer = build_tokenizer(args)
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    encoded_docs = pool.imap(encoder.encode, fin, 25)
    #encoded_docs = map(encoder.encode, fin)

    level = "document"
    if args.split_sentences:
        level = "sentence"

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Output prefix: {args.output_prefix}")
    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    for key in args.json_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix,
                                                      key, level)
        output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix,
                                                      key, level)
        builders[key] = indexed_dataset.make_builder(output_bin_files[key],
                                               impl=args.dataset_impl,
                                               vocab_size=tokenizer.vocab_size)
    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time to startup:", startup_end - startup_start)

    for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed
        for key, sentences in doc.items():
            
            if len(sentences) == 0:
                continue
            for sentence in sentences:
                builders[key].add_item(torch.IntTensor(sentence))
            builders[key].end_document()
        
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed/elapsed/1024/1024
            print(f"Processed {i} documents",
                  f"({i/elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)

    for key in args.json_keys:
        builders[key].finalize(output_idx_files[key])

if __name__ == '__main__':
    main()