import json
import pickle
# python -m spacy download en_core_web_sm
import spacy
spacy.load('en_core_web_sm')
nlp = spacy.load("en_core_web_sm")

from nltk.tokenize import sent_tokenize, word_tokenize

import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from collections import defaultdict, Counter
from nltk.corpus import stopwords
import random
import numpy as np
import re

# from multiprocessing import Pool as ProcessPool
import multiprocessing as mp
import itertools


from fever_athene.src.retrieval.fever_doc_db import FeverDocDB

DB = FeverDocDB(path = "/home/dcg-adlr-wping-source/nayeon/db/kilt_db.db")
DB_ABSTRACT= FeverDocDB(path = "/home/dcg-adlr-wping-source/nayeon/db/fever.db")
# DB = FeverDocDB(path = "/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/db/kilt_db.db")
# DB_ABSTRACT= FeverDocDB(path = "/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/db/fever.db")


STOPWORDS=set(stopwords.words('english'))

'''
DATA FORMAT:
    {"text": "bites into a melon\n\nwhen you wash your mouth\n\n* wipes his mouth *\n\nBlitz: That melon tastes so good!Dash: Your lips sure tastes better...Blitz: What?!Dash: *blushes* N-nothing... *Blitz: Well, if you want to taste my lips you can *looks at her*Dash: Maybe. You have a mouth full of melon. *giggles*Blitz:...And now?Dash: *blushes*-------------------Idea byDialog byand me", "adlr_id": "adlr-openwebtext2-0004235634", "split_id": "squad-drop-race-coqa-lambada-0003986376-0000-natural_questions-triviaqa-0003986397-0000-piqa-webqa-0003986783-0000", "score": {"toxicity": 0.6207597, "severe_toxicity": 0.4847491, "sexually_explicit": 0.8473677, "threat": 0.2560141, "profanity": 0.5090763, "identity_attack": 0.12999137}}
'''

def obtain_all_wiki_page_names():
    wiki_page_dir = '/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/data/wiki-pages'
    file_names = [f for f in listdir(wiki_page_dir) if isfile(join(wiki_page_dir, f))]

    all_wiki_names_set = []
    for file_name in tqdm(file_names):
        path = '{}/{}'.format(wiki_page_dir, file_name)
        with open(path) as infile:
            wiki_names = [json.loads(line)['id'].replace("_", ' ').replace('-LRB-', '( ').replace('-RRB-', ' )') 
                            for line in infile]

        all_wiki_names_set.extend(wiki_names)

    with open('/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/data/wiki_page_names.pickle', 'wb') as f:
        pickle.dump(all_wiki_names_set, f)

    print("Obatined {} wiki names...".format(len(all_wiki_names_set)))

def group_wiki_names():

    # 0. load list of wikinames
    with open('/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/data/wiki_page_names.pickle','rb') as f:
        all_wiki_names = pickle.load(f)

    wiki_nei_dict = defaultdict(list)
    
    print("Start building dict")
    for idx, wiki in tqdm(enumerate(all_wiki_names), total=len(all_wiki_names)):
        if "disambiguation" in wiki:
            continue 
        if "List of " in wiki:
            continue

        if "(" in wiki:
            wiki = wiki.split("(")[0]
        wiki_tokens = wiki.replace("( ","").replace(" )","").split(" ")
        
        if len(set(wiki_tokens).intersection(STOPWORDS)) == 0:
            if len(wiki_tokens) <= 5:
                for t in wiki_tokens:
                    wiki_nei_dict[t].append(wiki)


    print("Start pruning dict of size {}".format(len(wiki_nei_dict)))
    cnt = 0
    og_keys = list(wiki_nei_dict.keys())
    for key in tqdm(og_keys, total=len(og_keys)):
        if len(wiki_nei_dict[key]) == 1:
            del wiki_nei_dict[key]
            cnt+=1
    
    print("Pruned {} entry".format(cnt))

    with open('/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/data/wiki_to_nei_dict_5_tokens.pickle', 'wb') as f:
        pickle.dump(wiki_nei_dict, f)
    print("Successfully saved the dict of size {}!".format(len(wiki_nei_dict)))



def run_data_processing_in_parallel():
    
    with open('/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/data/wiki_to_nei_dict_3_tokens.pickle','rb') as f:
        wiki_to_nei = pickle.load(f)

    # 1. load abstract-only wikipedia dumps
    wiki_page_dir = '/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/data/wiki-pages'
    file_names = [f for f in listdir(wiki_page_dir) if isfile(join(wiki_page_dir, f))]
    # file_names = ['wiki-010.jsonl'] # debug

    workers = mp.Pool(processes=mp.cpu_count()) # num_workers=32. not provide number of worker makes it use all available cores

    ## version 2: write into N files
    with tqdm(total=len(file_names)) as pbar:
        for _ in tqdm(workers.imap_unordered(data_process_wrapper, zip(file_names, itertools.repeat(wiki_to_nei)))):
            pbar.update()
    
    ## version 1: write into ONE file
    # with tqdm(total=len(file_names)) as pbar:
    #     with open('/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/data/processed_wiki_dump.jsonl', 'w') as outfile:
    #         for update_text_objs in tqdm(workers.imap_unordered(data_process, file_names)):
    #             for obj in update_text_objs:
    #                 json.dump(obj, outfile)
    #                 outfile.write("\n")
    #             pbar.update()

def data_process_wrapper(_args):
    return data_process(*_args)

def data_process(filename, wiki_to_nei):
    # # 0. load list of wikinames
    # with open('/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/data/wiki_to_nei_dict_3_tokens.pickle','rb') as f:
    #     wiki_to_nei = pickle.load(f)

    with open('/home/nayeonl/megatron-lm/fever_dev_wiki_names.txt') as fever_infile:
        must_wiki_list = set(fever_infile.readlines())
    with open('/home/nayeonl/megatron-lm/fever_train_wiki_names.txt') as fever_infile:
        must_wiki_list.update(set(fever_infile.readlines()))


    wiki_to_nei_keys = set(wiki_to_nei.keys())

    wiki_page_dir = '/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/data/wiki-pages'
    ex_cnt, in_cnt = 0, 0

    path = '{}/{}'.format(wiki_page_dir, filename)
    out_path = '/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/data/wiki-pages-nonfactual/{}'.format(filename)
    # out_path = '/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/data/wiki-pages-nonfactual-v2/{}'.format(filename)
    # out_path = '/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/data/wiki-page-fever-dev/{}'.format(filename)

    start_idx = 0

    if os.path.isfile(out_path):
        start_idx = 20000 #sum(1 for line in open(out_path, 'r'))

    modified_wiki_texts = []

    write_every = 1000
    with open(path) as infile:
        
        all_json_obj = [ json.loads(line) for line in infile]
        for idx, json_obj in tqdm(enumerate(all_json_obj), total=len(all_json_obj)):

            if idx < start_idx:
                continue 
            
            if "disambiguation" in json_obj['id']:
                # skip any disambiguation pages
                continue

            if json_obj['text'] == '':
                continue
            
            text = json_obj['text']
            if 'refer to :' in text:
                # skip any disambiguation pages
                continue
            
            source_wiki = json_obj['id']

            # # for only focusing on wiki covering FEVER
            # source_wiki = source_wiki.replace("_", ' ').replace('-LRB-', '(').replace('-RRB-', ')')
            # if source_wiki not in must_wiki_list:
            #     continue

            if set(source_wiki.split(" ")).intersection(wiki_to_nei_keys): # exists_similiar_entity
                # do extrinsic
                ex_cnt += 1

                target_wiki_candidates = wiki_to_nei[source_wiki]
                text = re.sub(source_wiki, lambda m: random.choice(target_wiki_candidates), text)
            else:
                # do intrinsic
                in_cnt += 1

                doc = nlp(text)
                person_ne = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
                org_ne = [ent.text for ent in doc.ents if ent.label_ == 'ORG']

                if len(person_ne) < 2 and len(org_ne) < 2:
                    # no replacement of ne will happen --> factual text --> so, we skip this sample
                    continue 

                text = ne_replacement(text, person_ne)
                text = ne_replacement(text, org_ne)

            text = text.replace("_", ' ').replace('-LRB-', '(').replace('-RRB-', ')').replace('-LSB-', '[').replace('-RSB-', ']')
            obj = {'text': text, 'id': json_obj['id'], 'fname': filename, 'row_idx': idx}
            modified_wiki_texts.append(obj)

            if idx % write_every == 0:
                # First, write 
                for _obj in modified_wiki_texts:
                    with open(out_path, 'a') as outfile:
                        json.dump(_obj, outfile)
                        outfile.write("\n")

                # Then, flush
                modified_wiki_texts = []

    return modified_wiki_texts


def main():
    print("main")

    # 0. load list of wikinames
    with open('/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/data/wiki_to_nei_dict_3_tokens.pickle','rb') as f:
        wiki_to_nei = pickle.load(f)
    
    with open('/home/nayeonl/megatron-lm/fever_dev_wiki_names.txt') as fever_infile:
        must_wiki_list = set(fever_infile.readlines())
    with open('/home/nayeonl/megatron-lm/fever_train_wiki_names.txt') as fever_infile:
        must_wiki_list.update(set(fever_infile.readlines()))

    print(len(must_wiki_list))
    # exit(0)
    # 1. load abstract-only wikipedia dumps
    wiki_page_dir = '/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/data/wiki-pages'
    file_names = [f for f in listdir(wiki_page_dir) if isfile(join(wiki_page_dir, f))]
    # file_names = ['wiki-010.jsonl'] # debug

    ex_cnt, in_cnt = 0, 0

    for f_name in tqdm(file_names, total=len(file_names)):
        path = '{}/{}'.format(wiki_page_dir, f_name)

        modified_wiki_texts = []
        with open(path) as infile:
            
            all_json_obj = [ json.loads(line) for line in infile]

            for idx, json_obj in tqdm(enumerate(all_json_obj), total=len(all_json_obj)):

                if json_obj['text'] == '':
                    continue
                
                text = json_obj['text']
                if 'refer to :' in text:
                    # skip any disambiguation pages
                    continue
                
                source_wiki = json_obj['id']

        
                if set(source_wiki.split(" ")).intersection(wiki_to_nei.keys()): # exists_similiar_entity
                    # do extrinsic
                    ex_cnt += 1

                    target_wiki_candidates = wiki_to_nei[source_wiki]
                    text = re.sub(source_wiki, lambda m: random.choice(target_wiki_candidates), text)
                else:
                    # do intrinsic
                    in_cnt += 1

                    og_text = text

                    doc = nlp(text)
                    person_ne = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
                    org_ne = [ent.text for ent in doc.ents if ent.label_ == 'ORG']

                    if len(person_ne) < 2 and len(org_ne) < 2:
                        # no replacement of ne will happen --> factual text --> so, we skip this sample
                        continue 
                    text = ne_replacement(text, person_ne)
                    text = ne_replacement(text, org_ne)

                text = text.replace("_", ' ').replace(' -LRB- ', ' ( ').replace(' -RRB- ', ' ) ').replace(' -LSB- ', ' [ ').replace(' -RSB- ', ' ] ')
                modified_wiki_texts.append({'text': text, 'id': json_obj['id'], 'fname': f_name})


    print("EX: {}, IN: {}".format(ex_cnt, in_cnt))


def ne_replacement(text, nes):
    nes = list(set(nes))
    # og_text = text

    try:
        if len(nes) >= 2:
            for ne in nes: # by chance, there can be cases where original nes end up being not replaced. but low chance if len(nes) is relatively high
                # print([ 0.0 if ne_ == ne else 1.0/(len(nes)-1) for ne_ in nes])
                text = re.sub(ne, lambda m: list(np.random.choice(nes, 1, p=[ 0.0 if ne_ == ne else 1.0/(len(nes)-1) for ne_ in nes]))[0], text)

        elif len(nes) == 2:
            text = text.replace(nes[0], "==first_0==")
            text = text.replace(nes[1], nes[0])
            text = text.replace("==first_0==", nes[1])
            # print("[BEFORE]", og_text, "\n")
            # print("[AFTER]", text, "\n\n")
    except:
        text = text

    return text


def combine_all_wiki_jsonls(wiki_page_dir, all_out_path):
    file_names = [f for f in listdir(wiki_page_dir) if isfile(join(wiki_page_dir, f))]

    for file in tqdm(file_names):
        f_path = '{}/{}'.format(wiki_page_dir, file)
        json_lines = [json.loads(line) for line in open(f_path, 'r')]

        for json_obj in json_lines:
            with open(all_out_path, 'a') as outfile:
                json.dump(json_obj, outfile)
                outfile.write("\n")


def prepare_subset_factual_wiki_jsonls(wiki_page_dir, all_out_path, cover_just_fever):
    file_names = [f for f in listdir(wiki_page_dir) if isfile(join(wiki_page_dir, f))]

    with open('/home/nayeonl/megatron-lm/fever_dev_wiki_names.txt') as fever_infile:
        must_wiki_list = set(fever_infile.readlines())
    with open('/home/nayeonl/megatron-lm/fever_train_wiki_names.txt') as fever_infile:
        must_wiki_list.update(set(fever_infile.readlines()))

    for file in tqdm(file_names):
        f_path = '{}/{}'.format(wiki_page_dir, file)
        json_lines = [json.loads(line) for line in open(f_path, 'r')]

        # max_idx = 16890
        for idx, json_obj in enumerate(json_lines):
            source_wiki = json_obj['id']

            if cover_just_fever:
                # for only focusing on wiki covering FEVER
                source_wiki = source_wiki.replace("_", ' ').replace('-LRB-', '(').replace('-RRB-', ')')
                if source_wiki not in must_wiki_list:
                    continue
            # else:
            #     if idx > max_idx: # chunk jsonl file at max_idx
            #         break

            with open(all_out_path, 'a') as outfile:
                json.dump(json_obj, outfile)
                outfile.write("\n")



def prepare_full_wiki_for_fever(all_out_path, preprocess=None, use_abstract_only=False):

    def clean_wiki(sent):
        sent = sent.replace("[[","")
        sent = sent.replace("]]","")

        return sent

    with open('/home/nayeonl/megatron-lm/fever_dev_wiki_names.txt') as fever_infile:
        must_wiki_list = set(fever_infile.readlines())
    with open('/home/nayeonl/megatron-lm/fever_train_wiki_names.txt') as fever_infile:
        must_wiki_list.update(set(fever_infile.readlines()))


    total_word_cnt = 0
    no_she_he = 0
    with open(all_out_path, 'w') as outfile:
        for idx, wiki in tqdm(enumerate(list(must_wiki_list)), total=len(must_wiki_list)):

            wiki_name = wiki.replace("\n","")
            lines = DB.get_doc_lines(wiki_name)

            if use_abstract_only:
                text = lines.split("Section::::")[0] # just get the first section -- which is Wiki Abstract
                # print(text, "\n===================================================================================\n")

            else:
                text = " ".join([ clean_wiki(sent) for sent in sent_tokenize(lines) if '::::' not in sent and "File:" not in sent ])

            if preprocess == 'wiki_pronoun_handle':
                doc = nlp(text)
                subjects = [token.text.lower() for token in doc if token.dep_ in ['nsubj', 'nsubjpass']]
                c = Counter(subjects).most_common(3)
                c_words = [tuple_[0] for tuple_ in c]
                
                if 'she' in c_words:
                    text = text.replace(' she ', ' {} '.format(wiki_name))
                    text = text.replace(' She ', ' {} '.format(wiki_name))
                    # TODO what about "her" / "her" ? 
                elif 'he' in c_words:
                    text = text.replace(' he ', ' {} '.format(wiki_name))
                    text = text.replace(' He ', ' {} '.format(wiki_name))
                    # TODO what about "his"/ "him" ?
                else:
                    # print(wiki_name)
                    # print(c)
                    no_she_he+=1

            elif preprocess == 'wiki_name_prefix':
                text = " ".join([ wiki_name + " ==> " + sent for sent in sent_tokenize(text)])

            # else:
            #     print("Are you sure about no preprocessing?")
            #     print("No preprocessing! Default option~")

                # exit(0)
        
            obj = {'text': text, 'id': idx, 'fname': wiki_name}
            
            total_word_cnt += len(text.split(" "))
        
            json.dump(obj, outfile)
            outfile.write("\n")

    print("Saved {} wiki with {} tokens".format(len(must_wiki_list), total_word_cnt))
    print("No She/He {}".format(no_she_he))



def prepare_full_wiki_WITHOUT_fever(all_out_path):

    def clean_wiki(sent):
        sent = sent.replace("[[","")
        sent = sent.replace("]]","")

        return sent

    with open('/home/nayeonl/megatron-lm/fever_dev_wiki_names.txt') as fever_infile:
        FEVER_wiki_list = set(fever_infile.readlines())
    with open('/home/nayeonl/megatron-lm/fever_train_wiki_names.txt') as fever_infile:
        FEVER_wiki_list.update(set(fever_infile.readlines()))

    all_wiki_names = DB.get_non_empty_doc_ids()

    with open(all_out_path, 'w') as outfile:

        for idx, wiki_name in tqdm(enumerate(list(all_wiki_names)), total=len(all_wiki_names)):

            if wiki_name not in FEVER_wiki_list:
                lines = DB.get_doc_lines(wiki_name)

                text = " ".join([ clean_wiki(sent) for sent in sent_tokenize(lines) if '::::' not in sent and "File:" not in sent ])
            
                obj = {'text': text, 'id': idx, 'fname': wiki_name}
                
                json.dump(obj, outfile)
                outfile.write("\n")

    print("Saved {} wiki".format(len(all_wiki_names)))




def count_lines(in_path):
    json_lines = [json.loads(line) for line in open(in_path, 'r')]
    print("LINE COUNT: ", len(json_lines))


if __name__ == '__main__':
    # obtain_all_wiki_page_names()
    # group_wiki_names()

    # main()

    ##########################################################################################
    #                             Prepare non-factual Wiki                                   #
    ##########################################################################################
    ## step 1: run NE replacement in parallel
    # run_data_processing_in_parallel()

    ## step 2: combine the processed jsonls into one file
    # wiki_page_dir = '/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/data/wiki-pages-nonfactual'
    # all_out_path = '/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/data/wiki-pages-nonfactual-all_b4.jsonl'
    # combine_all_wiki_jsonls(wiki_page_dir, all_out_path)

    ## additional: check the number of wiki instances
    # count_lines(all_out_path)

    ##########################################################################################
    #                               Prepare factual Wiki                                     #
    ##########################################################################################
    # wiki_page_dir = '/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/data/wiki-pages'
    # all_out_path = '/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/data/wiki-pages-factual-all.jsonl'
    # combine_all_wiki_jsonls(wiki_page_dir, all_out_path)

    # # partial factual Wiki
    # wiki_page_dir = '/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/data/wiki-pages'
    # all_out_path = '/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/data/wiki-pages-factual-all.jsonl'
    # combine_all_wiki_jsonls(wiki_page_dir, all_out_path)


    ##########################################################################################
    #                               Prepare Wiki For FEVER                                   #
    ##########################################################################################

    ################################### all wiki #############################################
    # # ### default version without any pre-processing
    # # wiki_page_dir = '/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/data/wiki-pages'
    # all_out_path = '/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/data/full-wiki-pages-FEVER-all.jsonl'
    # prepare_full_wiki_for_fever(all_out_path)

    ## processed version 1: append "wiki_name==>" in front of every sentence.
    wiki_page_dir = '/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/data/wiki-pages'
    all_out_path = '/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/data/full-wiki-pages-FEVER-all-wikiName-prefix.jsonl'
    prepare_full_wiki_for_fever(all_out_path, preprocess='wiki_name_prefix')

    # ### processed version 2: replace all pronouns with the corresponding WikiName 
    # # wiki_page_dir = '/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/data/wiki-pages'
    # all_out_path = '/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/data/full-wiki-pages-FEVER-all-pronounHandled.jsonl'
    # prepare_full_wiki_for_fever(all_out_path, preprocess='wiki_pronoun_handle')

    # # ################################### wiki abstract #############################################

    # ### processed version 1: append "wiki_name==>" in front of every sentence.
    # all_out_path = '/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/data/abstract-wiki-pages-FEVER-all-wikiName-prefix.jsonl'
    # prepare_full_wiki_for_fever(all_out_path, preprocess='wiki_name_prefix', use_abstract_only=True)


    # ##########################################################################################
    # #                Prepare Wiki NOT containing FEVER - For validation ppl                  #
    # ##########################################################################################
    # all_out_path = '/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/data/full-wiki-pages-nonFEVER-validation-set.jsonl'
    # prepare_full_wiki_WITHOUT_fever(all_out_path)


    ## ################################### save a smaller subset of the full-wiki-pages-nonFEVER-validation-set.jsonl because it is just too big   ###################################
    # from random import sample

    # with open(all_out_path, 'r') as infile:
    #     all_val_obj = [ json.loads(line)['text'] for line in infile]

    #     # sub_text = " ".join(sample(all_val_obj,100))
    #     # print(len(sub_text))

    #     sub_text = "\n".join(sample(all_val_obj,2000))

    #     with open('/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/data/wikitext-103/ours.nonFEVER.valid.tokens', 'w') as outfile:
    #         outfile.write(sub_text)

    #     print("Saved {} CHARS".format(len(sub_text)))


    # # ##########################################################################################
    # # #             Testing out wiki mask processing for data_process.sh script                #
    # # ##########################################################################################
    # with open('/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/data/wikitext-103/ours.nonFEVER.valid.tokens', 'r') as infile:
    #     text_lines = sent_tokenize(infile.read())
    
    # for idx, text in enumerate(text_lines):
    #     doc = nlp(text)
    #     nes = [(ent.text, ent.label_) for ent in doc.ents]
    #     useful_deps = [(token.text.lower(), token.dep_) for token in doc if token.dep_ in ['nsubj', 'nsubjpass', 'ROOT', 'dobj', 'pobj', 'compound']]
    #     all_deps = [(token.text.lower(), token.dep_) for token in doc]

    #     root_idx = [ token.dep_ for token in doc].index('ROOT')
    #     all_tokens = [token for token in doc]

    #     print("[TEXT] ", text)
    #     # print("[ALL DEP] ", all_deps, "\n")

    #     print("[NEs] ", nes)
    #     print("[DEP] ", useful_deps)

    #     print("[ROOT]", root_idx, all_tokens[:root_idx+1])
    #     print("\n")



    #     if idx == 10:
    #         break

    print("Yay!")

