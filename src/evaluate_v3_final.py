
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
import numpy as np
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from retriever import obtain_relevant_evidences, get_wiki_from_db
from factuality_metric import nli_metric, ner_metric, nli_metric_batch

from src.const import DATA_DIR, HOME_DIR, GEN_DIR
from src.claim_handling import obtain_important_ne, has_incorrect_style

'''
    obj: LM generation object
    prompt_wiki_names (list): Wikipedia list from the FEVER dataset (evidence)

'''
TYPES = {
    'NO_FACT': 1,
    'HAS_FACT': 2,
    'OFF_TOPIC': 3
}

def identify_sentence_type(claim_obj, wiki_names_txt):
    assigned_type = None

    # case 1: no facts -- i.e., no NE, incorrect_style, no SUBJECT
    if len(claim_obj['important_ne']) + len(claim_obj['unimportant_ne']) == 0 or has_incorrect_style(claim_obj) or len(claim_obj['subject']) == 0:
        assigned_type = TYPES['NO_FACT']

    # case 2: no off-topic, but contains facts (unimportant_ne) about target-topic
    elif len(claim_obj['important_ne']) == 0 and len(claim_obj['unimportant_ne']) > 0:
        assigned_type = TYPES['HAS_FACT']

    # case 3: tricky scenario. important_ne could be relevant to the target-topic, or could indicate off-topic
    else:
        # 1. filter out any extra_ne that is same as wikiname -- e.g., wiki_name = Barak Obama, ne = Obama
        extra_ne = [ne[0] for ne in claim_obj['important_ne'] if ne[0] not in wiki_names_txt]

        # 2. check if any of the extra_ne is the "SUBJECT" of the generation
        overlap_between_extraNE_and_subj = claim_obj['subject'].intersection(set(" ".join(extra_ne).split(" ")))

        if len(overlap_between_extraNE_and_subj) > 0: # contains off-topic NE!!
            assigned_type = TYPES['OFF_TOPIC']
        else:
            assigned_type = TYPES['HAS_FACT']
    
    return assigned_type

def single_instance_eval(obj, prompt_wiki_names, run_nli_metric, run_ner_metric, first_sentence_only=True, verbose=False):

    wiki_names_txt = " ".join(prompt_wiki_names)

    text = obj['text'].strip()

    sent_type, claim_to_verify = "NO_GEN", ""
    hallu_ner_ratio = None
    nli_contradict_prob, nli_entail_prob, nli_neutral_prob, nli_label = None, None, None, None
    used_ev, evs = None, None
    top_10_evs_for_analysis = None

    # step 1: identify the type of the generation sent
    sents = sent_tokenize(text)
    if len(sents) != 0:

        gen_first_sent = sents[0]
        first_sent_obj_with_ne = obtain_important_ne(gen_first_sent.strip())

        sent_type = identify_sentence_type(first_sent_obj_with_ne, wiki_names_txt)
        claim_to_verify = first_sent_obj_with_ne['gen']

        # step 2: check factuality if gen contains fact
        
        if sent_type == TYPES['HAS_FACT']:

            wiki_sentences = get_wiki_from_db(prompt_wiki_names) # wiki_sentences from wiki_dump

            
            # evs = obtain_relevant_evidences(claim_to_verify, wiki_sentences, k=3, method='tfidf')
            top_10_evs_for_analysis = obtain_relevant_evidences(claim_to_verify, wiki_sentences, k=5, method='combined') # returns 10 sents
            evs  = obtain_relevant_evidences(claim_to_verify, wiki_sentences, k=1, method='combined') # will return 2 x k sentences

            if run_ner_metric:

                NE_to_check = first_sent_obj_with_ne['important_ne'] + first_sent_obj_with_ne['unimportant_ne']

                correct_ner_ratio = ner_metric(NE_to_check, wiki_sentences) # apply directly on wiki
                
                hallu_ner_ratio = 1 - correct_ner_ratio

            if run_nli_metric: 
                
                # # identify the evs that give highest nli entailment score
                premise_hypothesis_pairs = [[ev[0], claim_to_verify] for ev in evs]
                nli_probs, labels = nli_metric_batch(premise_hypothesis_pairs)

                entailment_argmax = np.argmax([nli_s[2] for nli_s in nli_probs])
                max_prob = nli_probs[entailment_argmax]
                max_label = labels[entailment_argmax]
                used_ev = evs[entailment_argmax]

        
                # print(max_prob, max_label)
                nli_contradict_prob = max_prob[0] 
                nli_neutral_prob = max_prob[1] 
                nli_entail_prob = max_prob[2]

                nli_label = max_label



    eval_result_obj = {
        'claim_type':sent_type,
        'claim_to_verify': claim_to_verify,
        'hallu_ner': hallu_ner_ratio,
        'nli-contr': nli_contradict_prob,
        'nli-entail': nli_entail_prob,
        'nli-neutr': nli_neutral_prob,
        'nli-label': nli_label,
        'used_ev': used_ev,
        'evs': evs,
        'top10': top_10_evs_for_analysis
    }

    return eval_result_obj
    

def main(args):

    prompt_type = args.prompt_type # 'factual' 

    run_nli_metric = True
    run_ner_metric = True

    prompt_path = '{}/prompts/fever_{}_final.jsonl'.format(HOME_DIR, prompt_type)
    if args.gen_path != None:
        gen_path = '{}/{}'.format(GEN_DIR, args.gen_path)
    else:
        print("No generation path provided. Using template based path")
        exit(0)


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

    final_hallu_ner_score, final_contradict_prob, final_neutral_prob, final_entail_prob = 0, 0, 0, 0
    final_strict_true_ner_score = 0

    no_fact_cnt, has_fact_cnt, off_topic_cnt = 0,0,0

    all_nli_labels = []

    no_wiki_cnt = 0

    lm_nonfactual_analysis_list = []
    all_analysis_list = []

    assert len(prompts) == len(gens)

    for _, (prompt_obj, gen_obj) in tqdm(enumerate(zip(prompts, gens)), total=len(prompts)):

        prompt_wiki_names = [ev_infos[0] for ev_infos in prompt_obj['evidence_info']]
        
        res_obj = single_instance_eval(gen_obj, prompt_wiki_names, run_nli_metric, run_ner_metric)

        if res_obj['claim_type'] == TYPES['NO_FACT']:
            no_fact_cnt += 1

        elif res_obj['claim_type'] == TYPES['OFF_TOPIC']:
            off_topic_cnt += 1

        elif res_obj['claim_type'] == TYPES['HAS_FACT']:
            # print("\n[PROMPT {}] {}".format(idx, gen_obj['prompt']))
            # print("[LM continuation {}] {}".format(idx, gen_obj['text']))
            # print("[CHECKWORTHY {}] {}".format(idx, res_obj['claim_to_verify']))

            has_fact_cnt += 1

            final_hallu_ner_score += res_obj['hallu_ner']
            final_contradict_prob += res_obj['nli-contr']
            final_neutral_prob += res_obj['nli-neutr']
            final_entail_prob += res_obj['nli-entail']
            all_nli_labels.append(res_obj['nli-label'])

            all_analysis_list.append(
                {
                    'wiki': " ".join(prompt_wiki_names),
                    'prompt': gen_obj['prompt'],
                    'lm-gen': res_obj['claim_to_verify'],
                    'hallu_ner': res_obj['hallu_ner'],
                    'nli-label': res_obj['nli-label'],
                    'nli-entail': res_obj['nli-entail'],
                    'nli-contr': res_obj['nli-contr'],
                    'nli-neutral': res_obj['nli-neutr'],
                    'used_ev': res_obj['used_ev'],
                    'top10': res_obj['top10']
                }
            )

            if run_ner_metric and run_nli_metric:
                # strict form of metric
                if res_obj['hallu_ner'] == 1.0 and res_obj['nli-label'] == 2:
                    # No hallucinated NER AND NLI class = support
                    # strictly_correct_cnt += 1

                    final_strict_true_ner_score += 1

    # analysis
    if args.save_gen_for_analysis:
        analysis_save_path = gen_path.replace(".jsonl", "_allGen.csv")
        
        df = pd.DataFrame(all_analysis_list)
        df.to_csv(analysis_save_path)

    total_cnt = no_fact_cnt + has_fact_cnt + off_topic_cnt

    no_fact_ratio=no_fact_cnt/total_cnt
    has_fact_ratio=has_fact_cnt/total_cnt
    off_topic_ratio=off_topic_cnt/total_cnt

    avg_hallu_ner_ratio = final_hallu_ner_score/has_fact_cnt
    avg_strict_true_ner = final_strict_true_ner_score/has_fact_cnt

    avg_contradict_prob = final_contradict_prob/has_fact_cnt
    avg_neutral_prob = final_neutral_prob/has_fact_cnt
    avg_entail_prob = final_entail_prob/has_fact_cnt

    print("NO_FACT: {:.2f}%, HAS_FACT: {:.2f}%, OFFTOPIC: {:.2f}%".format(no_fact_ratio*100, has_fact_ratio*100, off_topic_ratio*100))

    print("Hallu NER: {:.2f}% | Strict True NER: {:.2f}%".format(avg_hallu_ner_ratio*100, avg_strict_true_ner*100))
    print("AVG PROBS: Contradict: {:.2f}%, Neutral: {:.2f}%, Entail: {:.2f}%".format(avg_contradict_prob*100, avg_neutral_prob*100, avg_entail_prob*100))

    nli_contradict_class_ratio, nli_neutral_class_ratio, nli_entail_class_ratio = 0, 0, 0

    if run_nli_metric:
        nli_counter = Counter(all_nli_labels)

        nli_contradict_class_ratio=nli_counter[0]/(nli_counter[0]+nli_counter[1]+nli_counter[2])
        nli_neutral_class_ratio=nli_counter[1]/(nli_counter[0]+nli_counter[1]+nli_counter[2])
        nli_entail_class_ratio=nli_counter[2]/(nli_counter[0]+nli_counter[1]+nli_counter[2])
        
        print("NLI CLASS %: Contradict: {:.2f}%, Neutral: {:.2f}%, Entail: {:.2f}%".format(
            nli_contradict_class_ratio*100,
            nli_neutral_class_ratio*100,
            nli_entail_class_ratio*100
        ))


    res_path = gen_path.replace(".jsonl", "_results.jsonl")
    with open(res_path, 'a') as outfile:
        res_obj = {
            "avg_hallu_ner_ratio": avg_hallu_ner_ratio,
            "nli_contradict_class_ratio": nli_contradict_class_ratio,
            "nli_neutral_class_ratio": nli_neutral_class_ratio, 
            "nli_entail_class_ratio": nli_entail_class_ratio,
            "no_fact_ratio": no_fact_ratio, 
            "has_fact_ratio": has_fact_ratio, 
            "off_topic_ratio": off_topic_ratio
        }
        json.dump(res_obj, outfile)
        outfile.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--prompt_type', type=str, help='name of prompt type of the testset [factual, nonfactual]')
    parser.add_argument('--exp_name', type=str, default='', help='experiment name for different generation techniques') 
    parser.add_argument('--debug_sample_size', type=int, default=None, help='# of sample size to use for debugging purpose. providing this value will automatically lead to debug mode')
    parser.add_argument('--gen_path', type=str, default=None, help='path to generations to evaluate') 

    parser.add_argument('--save_gen_for_analysis', action='store_true', help='Flag for saving some lm-gens with its metric for analysis') 



    args = parser.parse_args()
    main(args)

    print("yay!")