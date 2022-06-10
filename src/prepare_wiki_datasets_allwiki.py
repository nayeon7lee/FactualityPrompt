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

# DB = FeverDocDB(path = "/home/dcg-adlr-wping-source/nayeon/db/kilt_db.db")
DB = FeverDocDB(path = "/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/db/kilt_db.db")


STOPWORDS=set(stopwords.words('english'))

'''
DATA FORMAT:
    {"text": "bites into a melon\n\nwhen you wash your mouth\n\n* wipes his mouth *\n\nBlitz: That melon tastes so good!Dash: Your lips sure tastes better...Blitz: What?!Dash: *blushes* N-nothing... *Blitz: Well, if you want to taste my lips you can *looks at her*Dash: Maybe. You have a mouth full of melon. *giggles*Blitz:...And now?Dash: *blushes*-------------------Idea byDialog byand me", "adlr_id": "adlr-openwebtext2-0004235634", "split_id": "squad-drop-race-coqa-lambada-0003986376-0000-natural_questions-triviaqa-0003986397-0000-piqa-webqa-0003986783-0000", "score": {"toxicity": 0.6207597, "severe_toxicity": 0.4847491, "sexually_explicit": 0.8473677, "threat": 0.2560141, "profanity": 0.5090763, "identity_attack": 0.12999137}}
'''


def clean_wiki(sent):
    sent = sent.replace("[[","")
    sent = sent.replace("]]","")

    return sent

def main():

    # raw_wiki_path = '/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/KILT/kilt_knowledgesource.json'
    
    processed_wiki_path = '/home/dcg-adlr-wping-source/nayeon/data/full-wiki-pages-all-wikiName-prefix.jsonl'

    all_wiki_names = DB.get_non_empty_doc_ids()

    with open(processed_wiki_path, 'w') as outfile:
        for idx, wiki_name in tqdm(enumerate(list(all_wiki_names)), total=len(all_wiki_names)):

            lines = DB.get_doc_lines(wiki_name)

            text = " ".join([ 
                                wiki_name + " ==> " + clean_wiki(sent) 
                                for sent in sent_tokenize(lines) 
                                if '::::' not in sent and "File:" not in sent 
                            ])
        

            obj = {'text': text, 'id': idx, 'fname': wiki_name}
            
            json.dump(obj, outfile)
            outfile.write("\n")

    print("Saved {} wiki".format(len(all_wiki_names)))



if __name__ == '__main__':
    main()
    print("Yay!")

