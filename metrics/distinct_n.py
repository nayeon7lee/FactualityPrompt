import argparse
import json
import os
import logging
from collections import Counter

from os import listdir
from os.path import isfile, join
from tqdm import tqdm

from nltk.tokenize import word_tokenize, sent_tokenize

import multiprocessing
import itertools
import numpy as np

# from fever_athene.src.retrieval.fever_doc_db import FeverDocDB

logger = logging.getLogger(__name__)

'''
    Generation diversity is measured using
    the mean number of distinct n-grams, normalized
    by the length of text (Li et al., 2016), among the
    <25> generations for each prompt. We report Dist-1,
    Dist-2, and Dist-3 scores for distinct uni-, bi-, and
    trigrams, respectively
'''

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", type=str)
    parser.add_argument("--file_template", type=str)
    # parser.add_argument("N", type=int, help="N in distinct-N metric")
    parser.add_argument("--numbers-only", action="store_true")
    return parser.parse_args()


def wiki_distinct_n(n, factual_examples):
    counter = Counter()
    distinct_set = set()
    n_total = 0
    n_distinct = 0

    for example in tqdm(factual_examples, total=len(factual_examples)):
        if example.strip() != "":
            gen_tokens = word_tokenize(example.strip())
            for token in zip(*(gen_tokens[i:] for i in range(n))):
                distinct_set.add(token)
                # print("if", token)
                # if token not in counter:
                #     n_distinct += 1
                #     print("if", token)
                # elif counter[token] == 1:
                #     n_distinct -= 1
                #     print("elif", token)
                # else:
                #     print("enter 'else'")
                # counter[token] += 1
                n_total += 1

    return len(distinct_set), n_total, n


def distinct_n(n, factual_examples, f_name_template):
    counter = Counter()
    distinct_set = set()
    n_total = 0
    n_distinct = 0

    for example in tqdm(factual_examples, total=len(factual_examples)):
        if example['text'].strip() != "":
            use_first_sent_only=True
            if use_first_sent_only:
                gen = sent_tokenize(example['text'])[0] 
            else:
                gen = example['text']


            if "WikiNamePrefix" in f_name_template:
                wikiPrefix = example['prompt'].split(". ")[-1].strip()
                gen = gen.replace(wikiPrefix, " ")

            gen_tokens = word_tokenize(gen)
            for token in zip(*(gen_tokens[i:] for i in range(n))):
                distinct_set.add(token)
                # print("if", token)
                # if token not in counter:
                #     n_distinct += 1
                #     print("if", token)
                # elif counter[token] == 1:
                #     n_distinct -= 1
                #     print("elif", token)
                # else:
                #     print("enter 'else'")
                # counter[token] += 1
                n_total += 1

    return len(distinct_set), n_total, n


# def mp_distinct_n(factual_examples, n, f_name_template):
#     counter = Counter()
#     n_total = 0
#     n_distinct = 0

#     p = multiprocessing.Pool() # use all available thread
#     p.map(_helper, factual_examples)

#     def _helper(example):
#         if example['text'].strip() != "":
#             use_first_sent_only=True
#             if use_first_sent_only:
#                 gen = sent_tokenize(example['text'])[0] 
#             else:
#                 gen = example['text']

#             if "WikiNamePrefix" in f_name_template:
#                 wikiPrefix = example['prompt'].split(". ")[-1].strip()
#                 gen = gen.replace(wikiPrefix, " ")

#             gen_tokens = word_tokenize(gen)
#             for token in zip(*(gen_tokens[i:] for i in range(n))):
                
#                 if token not in counter:
#                     n_distinct += 1
#                 elif counter[token] == 1:
#                     n_distinct -= 1
#                 counter[token] += 1
#                 n_total += 1

#     return n_distinct, n_total

def main_dev():
    args = parse_args()

    dir = args.gen_dir
    f_template = args.file_template

    target_files = [f for f in listdir(dir) if isfile(join(dir, f)) and f_template in f]

    factual_examples = []
    for target_file in target_files:
        # print("{}/{}".format(dir, target_file))
        with open("{}/{}".format(dir, target_file), "r") as fin:
            # for l in fin:
            #     first_line = json.loads(l.strip())
            #     print(sent_tokenize(first_line['text'])[0] )
            #     break
            factual_examples.extend([json.loads(l.strip()) for l in fin])


    out_log_path = "distinct_n.log"
    with open(out_log_path, "a") as outfile:
        outfile.write(f"{f_template}")

    for n in [4, 3, 2]:
        # n_distinct, n_total = distinct_n(factual_examples, args.N)
        n_distinct, n_total = distinct_n(factual_examples, n, f_template)

        if 'greedy' in f_template:
            n_total = n_total * 15 # greedy will always generate same. So test on just one generation file, and multiply by # of seed used (in our case, 15)

        print("N:{}, N_distinct:{}, N_total:{}, n_distinct/N_total:{}".format(n, n_distinct, n_total, n_distinct/n_total))

        with open(out_log_path, "a") as outfile:
            # outfile.write(f"{n_distinct}, {n_total}, {n_distinct/n_total}")
            outfile.write(", {}, {}, {}".format(n_distinct, n_total, n_distinct/n_total))

    with open(out_log_path, "a") as outfile:
        outfile.write("\n")


def main():
    args = parse_args()

    dir = args.gen_dir
    f_template = args.file_template

    print(f_template)

    factual_target_files = [f for f in listdir(dir) if isfile(join(dir, f)) and f_template in f and 'nonfactual' not in f]
    nonfactual_target_files = [f for f in listdir(dir) if isfile(join(dir, f)) and f_template in f and 'nonfactual' in f]

    print(len(factual_target_files))
    print(len(nonfactual_target_files))

    
    assert len(factual_target_files) >= 10
    assert len(nonfactual_target_files) >= 10

    factual_target_files = factual_target_files[:10]
    nonfactual_target_files = nonfactual_target_files[:10]


    factual_examples = []
    for target_file in factual_target_files:
        # print("{}/{}".format(dir, target_file))
        with open("{}/{}".format(dir, target_file), "r") as fin:
            factual_examples.extend([json.loads(l.strip()) for l in fin])
    

    nonfactual_examples = []
    for target_file in nonfactual_target_files:
        # print("{}/{}".format(dir, target_file))
        with open("{}/{}".format(dir, target_file), "r") as fin:
            nonfactual_examples.extend([json.loads(l.strip()) for l in fin])


    # print(len(factual_examples), len(nonfactual_examples))
    # assert len(factual_examples) == 80000
    # assert len(nonfactual_examples) == 80000

    
    out_log_path = "/home/nayeonl/megatron-lm/final_distinct_n.log"
    with open(out_log_path, "a") as outfile:
        outfile.write(f"{f_template}")

    # factual examples
    workers = multiprocessing.Pool(4)

    factual_res_dict = {}
    for (_n_distinct, _n_total, _n) in workers.imap_unordered(distinct_n_wrapper, zip([1,2,3,4], itertools.repeat(factual_examples), itertools.repeat(f_template))):
        
        if 'greedy' in f_template:
            _n_total = _n_total * 10 # greedy will always generate same. So test on just one generation file, and multiply by # of seed used (in our case, 10)

        print(_n, _n_distinct, _n_total)
        factual_res_dict[_n] = float(_n_distinct/_n_total)

    with open(out_log_path, "a") as outfile:
        for n in [4,3,2]:
            outfile.write(", {}".format(factual_res_dict[n]))

    nonfactual_res_dict = {}
    for (_n_distinct, _n_total, _n) in workers.imap_unordered(distinct_n_wrapper, zip([1,2,3,4], itertools.repeat(nonfactual_examples), itertools.repeat(f_template))):
        
        if 'greedy' in f_template:
            _n_total = _n_total * 10 # greedy will always generate same. So test on just one generation file, and multiply by # of seed used (in our case, 10)

        print(_n, _n_distinct, _n_total)
        nonfactual_res_dict[_n] = float(_n_distinct/_n_total)

    with open(out_log_path, "a") as outfile:
        for n in [4,3,2]:
            outfile.write(", {}".format(nonfactual_res_dict[n]))

    with open(out_log_path, "a") as outfile:
        outfile.write("\n")


def distinct_n_wrapper(_args):
    return distinct_n(*_args)
    
def wiki_distinct_n_wrapper(_args):
    return wiki_distinct_n(*_args)

# def main_wiki_pages():
    
#     wiki_page_examples = []

#     DB = FeverDocDB(path = "/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/db/kilt_db.db")

#     with open('/home/nayeonl/megatron-lm/fever_dev_wiki_names.txt') as fever_infile:
#         fever_dev_wiki_names = set(fever_infile.readlines())

#     with open('/home/nayeonl/megatron-lm/fever_train_wiki_names.txt') as fever_infile:
#         fever_dev_wiki_names.update(set(fever_infile.readlines()))

#         for idx, wiki in tqdm(enumerate(fever_dev_wiki_names), total=len(fever_dev_wiki_names)):
#             wiki_name = wiki.replace("\n","")
#             lines = sent_tokenize(DB.get_doc_lines(wiki_name).replace("\n", " "))
#             wiki_page_examples.extend(lines)

#             if idx == 5000:
#                 break


#     print("wiki sent cnt", len(wiki_page_examples), "wiki doc #", len(fever_dev_wiki_names))
#     workers = multiprocessing.Pool(4)

#     factual_res_dict = {}
#     for (_n_distinct, _n_total, _n) in workers.imap_unordered(wiki_distinct_n_wrapper, zip([1,2,3,4], itertools.repeat(wiki_page_examples))):
        
#         # print(_n, _n_distinct, _n_total)
#         factual_res_dict[_n] = float(_n_distinct/_n_total)

#     for n in [4,3,2]:
#         print(", {}".format(factual_res_dict[n]))


if __name__ == '__main__':
    main()
    # main_wiki_pages()

    # n_distinct, n_total = distinct_n(factual_examples[:1000], 2, f_template)

    # workers = multiprocessing.Pool() # use all available thread

    # n_distinct_2, n_total_2= 0, 0
    # factual_examples_chunks = [chunk_np.tolist() for chunk_np in np.array_split(factual_examples[:1000], 20)]

    # for (_n_distinct, _n_total) in workers.imap_unordered(distinct_n_wrapper, zip(factual_examples_chunks,itertools.repeat(2), itertools.repeat(f_template))):
    #     n_distinct_2 += _n_distinct
    #     n_total_2 += _n_total

    # print(n_distinct, n_total)
    # print(n_distinct_2, n_total_2)

