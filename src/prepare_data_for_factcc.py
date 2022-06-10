
from ipaddress import _BaseAddress
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from collections import Counter
import re
import json
import argparse

import spacy
# spacy.prefer_gpu()
spacy.load('en_core_web_sm')
nlp = spacy.load("en_core_web_sm")

import pandas as pd
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from retriever import obtain_wiki, obtain_relevant_evidences, get_wiki_from_db
from metric import nli_metric, ner_metric

# DATA_DIR = '/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl'
# HOME_DIR = '/home/nayeonl/megatron-lm/'
from src.const import DATA_DIR, HOME_DIR
from src.claim_handling import obtain_important_ne, has_incorrect_style



def obtain_claim_wikitext_pair(obj, prompt_wiki_names, run_nli_metric, run_ner_metric, test_og_fever=False, first_sentence_only=True, verbose=False):

    wiki_names_txt = " ".join(prompt_wiki_names)

    if test_og_fever:
        text = obj['prompt'].strip() # NOTE: for evaluating against FEVER claims -- for testing how good our metric is
    else:
        text = obj['text'].strip() # NOTE: OG code for testing the generation

    gens = sent_tokenize(text)
    gens_with_ne = [obtain_important_ne(gen.strip()) for gen in gens]


    no_fact_gen_cnt, no_fact_gens = 0, []
    checkworthy_gen_cnt, checkworthy_gens = 0, []
    off_topic_gen_cnt, off_topic_gens = 0, []


    for claim_obj in gens_with_ne:
        
        # case 1: no facts -- i.e., no NE, incorrect_style, no SUBJECT
        if len(claim_obj['important_ne']) + len(claim_obj['unimportant_ne']) == 0 or has_incorrect_style(claim_obj) or len(claim_obj['subject']) == 0:
            no_fact_gen_cnt += 1
            # no_fact_gens.append(claim_obj['gen'])
            # print("[NO FACT]", claim_obj['gen'])

        # case 2: no off-topic, but contains facts (unimportant_ne) about target-topic
        elif len(claim_obj['important_ne']) == 0 and len(claim_obj['unimportant_ne']) > 0:
            checkworthy_gen_cnt += 1
            checkworthy_gens.append(claim_obj)
            # print('[CHECK-WORTHY]', claim_obj['gen'])

        # case 3: tricky scenario. important_ne could be relevant to the target-topic, or could indicate off-topic
        else:
            
            # 1. filter out any extra_ne that is same as wikiname -- e.g., wiki_name = Barak Obama, ne = Obama
            extra_ne = [ne[0] for ne in claim_obj['important_ne'] if ne[0] not in wiki_names_txt]

            # 2. check if any of the extra_ne is the "SUBJECT" of the generation
            overlap_between_extraNE_and_subj = claim_obj['subject'].intersection(set(" ".join(extra_ne).split(" ")))

            if len(overlap_between_extraNE_and_subj) > 0: # contains off-topic NE!!
                off_topic_gen_cnt += 1
                # off_topic_gens.append(claim_obj['gen'])
                # print('[OFF-TOPIC]', claim_obj['gen'])
            else:
                checkworthy_gen_cnt += 1
                checkworthy_gens.append(claim_obj)
                # print('[CHECK-WORTHY]', claim_obj['gen'])

    wiki_sentences = get_wiki_from_db(prompt_wiki_names) # wiki_sentences from wiki_dump

    # skip if there is i)  no wiki, or ii) no checkworthy claims.
    if wiki_sentences == [] or checkworthy_gens == []:
        return "", ""

    claim_to_verify = checkworthy_gens[0]['gen']
    top_k = obtain_relevant_evidences(claim_to_verify, wiki_sentences, k=10, method='tfidf')
    evidence_doc = [ev[0] for ev in top_k]

    # _, tfidf_evs, emb_evs = obtain_relevant_evidences(claim_to_verify, wiki_sentences, k=5, method='bertscore')

    # # combine top-5 sentences selected from best of both world
    # evidence_doc = [ev[0] for ev in tfidf_evs] + [ev[0] for ev in emb_evs]
    
    return claim_to_verify, " ".join(evidence_doc)


def main(args):

    model_size = args.model_size #'1.3b'
    # prompt_type = 'nonfactual' # 1500
    prompt_type = args.prompt_type # 'factual' 

    run_nli_metric = True
    run_ner_metric = True


    prompt_path = '{}/prompts/fever_dev_{}_v2.jsonl'.format(HOME_DIR, prompt_type)
    gen_path = '{}/generations/{}_{}_fever{}.jsonl'.format(DATA_DIR, model_size, prompt_type, args.exp_name)

    prompts, gens = [], []
    with open(prompt_path, 'r') as infile:
        for line in infile:
            fever_obj = json.loads(line.strip())
            prompts.append(fever_obj)

    with open(gen_path, 'r') as infile:
        for line in infile:
            gen_obj = json.loads(line.strip())
            gens.append(gen_obj)
    

    # DEBUG mode!
    if args.debug_sample_size != None:
        DEBUG_SAMPLE_SIZE = args.debug_sample_size #300

        prompts = prompts[:DEBUG_SAMPLE_SIZE]
        gens = gens[:DEBUG_SAMPLE_SIZE]

    res_path = '{}/factCC_temp/{}.{}.jsonl'.format(DATA_DIR, model_size, prompt_type)

    if args.test_og_fever:
        res_path = '{}/factCC_temp/fever.{}.jsonl'.format(DATA_DIR, prompt_type)
    else:
        res_path = '{}/factCC_temp/{}.{}.jsonl'.format(DATA_DIR, model_size, prompt_type)

    # res_path = '{}/factCC_temp/data-dev.jsonl'.format(DATA_DIR)
    for idx, (prompt_obj, gen_obj) in tqdm(enumerate(zip(prompts, gens)), total=len(prompts)):

        assert prompt_obj['prompt'] == gen_obj['prompt']

        prompt_wiki_names = [ev_infos[0] for ev_infos in prompt_obj['evidence_info']]

        claim, wiki_top10_texts = obtain_claim_wikitext_pair(gen_obj, prompt_wiki_names, run_nli_metric, run_ner_metric, args.test_og_fever)

        if claim != "" and wiki_top10_texts != "":
            with open(res_path, 'a') as outfile:
                obj = {
                    'id': prompt_obj['id'],
                    'claim': claim, 
                    'text': wiki_top10_texts,
                    'label': 'INCORRECT'
                }

                json.dump(obj, outfile)
                outfile.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--prompt_type', type=str, help='name of prompt type of the testset [factual, nonfactual]')
    parser.add_argument('--model_size', type=str, default='1.3b', help='LM model size')
    parser.add_argument('--exp_name', type=str, default='', help='additional experiment info for gen path')
    parser.add_argument('--debug_sample_size', type=int, default=None, help='# of sample size to use for debugging purpose. providing this value will automatically lead to debug mode')
    parser.add_argument('--test_og_fever', action='store_true', help='Evaluate fever claims with label') 


    args = parser.parse_args()
    main(args)

    print("yay!")