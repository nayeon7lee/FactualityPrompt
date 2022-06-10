
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

from retriever import obtain_relevant_evidences, get_wiki_from_db # obtain_wiki
from metric import nli_metric, ner_metric

# DATA_DIR = '/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl'
# HOME_DIR = '/home/nayeonl/megatron-lm/'
from src.const import DATA_DIR, HOME_DIR, GEN_DIR
from src.claim_handling import obtain_important_ne, has_incorrect_style

'''
    obj: LM generation object
    prompt_wiki_names (list): Wikipedia list from the FEVER dataset (evidence)

'''



def single_instance_eval(obj, prompt_wiki_names, run_nli_metric, run_ner_metric, test_og_fever=False, first_sentence_only=True, verbose=False):

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

    gen_type_cnts = {
        'NO_FACT': no_fact_gen_cnt,
        'CHECK_WORTHY': checkworthy_gen_cnt,
        'OFF_TOPIC': off_topic_gen_cnt
    }

    nli_contradict_prob, nli_entail_prob, nli_neutral_prob = 0, 0, 0
    correct_ner_ratio = 0

    nli_label = -1
    # wiki_sentences = obtain_wiki(prompt_wiki_names) # wiki_sentences from internet
    wiki_sentences = get_wiki_from_db(prompt_wiki_names) # wiki_sentences from wiki_dump

    # skip if there is i)  no wiki, or ii) no checkworthy claims.
    if wiki_sentences == [] or checkworthy_gens == []:
        # print("ATTENTION!! EMPTY WIKI!! FOR wiki: ", prompt_wiki_names)
        # with open("wiki_error.log", 'a') as outfile:
        #     outfile.write(" /// ".join(prompt_wiki_names)+"\n")
        return gen_type_cnts, None

    for claim_obj in checkworthy_gens:
        claim_to_verify = claim_obj['gen']
        
        # topk_evs = obtain_relevant_evidences(claim_to_verify, wiki_sentences, k=3, method='bertscore')
        topk_evs = obtain_relevant_evidences(claim_to_verify, wiki_sentences, k=3, method='tfidf')

        if run_ner_metric:
            
            ignore_target_wiki_ne = True
            if ignore_target_wiki_ne:
                NE_to_check = claim_obj['unimportant_ne']

                # filter out the target Wiki's NE (cuz, theoretically, all sentences will contain this NE. so meaningless)
                extra_ne = [ne[0] for ne in NE_to_check if ne[0] not in wiki_names_txt]

                for ent in claim_obj['important_ne']:
                    if any([bool(word in wiki_names_txt) for word in ent[0].split(" ")]):
                        continue
                    else:
                        NE_to_check.append(ent)
                        
                if len(NE_to_check) == 0: # ignore samples with 0 extra NE
                    return gen_type_cnts, None
            else:
                NE_to_check = claim_obj['important_ne']+claim_obj['unimportant_ne']

            _correct_ner_ratio = ner_metric(NE_to_check, wiki_sentences) # apply directly on wiki

            # if _correct_ner_ratio < 1.0:
            #     print("GEN", claim_to_verify)
            #     print("Correct NER", _correct_ner_ratio, NE_to_check)
            correct_ner_ratio += _correct_ner_ratio

        if run_nli_metric:
            # TODO for now im using top-1 evidence, but i think it is more robust to use top-2. but how? 
            top1_evidence = topk_evs[0][0]
            nli_scores = nli_metric(premise=top1_evidence, hypothesis=claim_to_verify) # [[contradiction, neutral, entailment], argmax]
            
            
            nli_contradict_prob += nli_scores[0][0][0] 
            nli_neutral_prob += nli_scores[0][0][1] 
            nli_entail_prob += nli_scores[0][0][2]
            
            # print("[WIKI_NAME] ", prompt_wiki_names)
            # print("[LM GEN] ", claim_obj['gen'])
            # print("[EVIDENCE] ", topk_evs)
            # print("[NLI] Contradict: {}, Neutral: {}, Entail: {}".format(nli_contradict_prob, nli_neutral_prob, nli_entail_prob))
            

        if first_sentence_only:
            # evaluate with only first checkworthy gen
            if run_nli_metric: nli_label = nli_scores[1]

            # if str(nli_label) == '2':
            #     print("[WIKI_NAME] ", prompt_wiki_names)
            #     print("[LM GEN] ", claim_obj['gen'])
            #     print("[EVIDENCE] ", topk_evs)
            #     print("[NLI] Contradict: {}, Neutral: {}, Entail: {}".format(nli_contradict_prob, nli_neutral_prob, nli_entail_prob))
            break 


    gen_cnt = len(checkworthy_gens) if len(checkworthy_gens) > 0 and not first_sentence_only else 1
    

    metrics = {
        'claim_verified': claim_to_verify,
        'ner': correct_ner_ratio/gen_cnt,
        'nli-contr': nli_contradict_prob/gen_cnt,
        'nli-entail': nli_entail_prob/gen_cnt,
        'nli-neutr': nli_neutral_prob/gen_cnt,
        'nli-label': nli_label
    }

    return gen_type_cnts, metrics
            



def main(args):

    model_size = args.model_size #'1.3b'
    prompt_type = args.prompt_type # 'factual' 

    run_nli_metric = True
    run_ner_metric = True

    prompt_path = '{}/prompts/fever_dev_{}_v2.jsonl'.format(HOME_DIR, prompt_type)
    if args.gen_path != None:
        gen_path = '{}/{}'.format(GEN_DIR, args.gen_path)
    else:
        print("No generation path provided. Using template based path")
        gen_path = '{}/generations/{}_{}_fever{}.jsonl'.format(DATA_DIR, model_size, prompt_type, args.exp_name)

    res_template = '{}/results/{}.{}'.format(DATA_DIR, model_size, prompt_type)

    prompts, gens = [], []
    with open(prompt_path, 'r') as infile:
        for line in infile:
            fever_obj = json.loads(line.strip())
            prompts.append(fever_obj)

    with open(gen_path, 'r') as infile:
        for line in infile:
            gen_obj = json.loads(line.strip())
            gens.append(gen_obj)
    
    res_path = res_template + '.metric.txt'
    outfile = open(res_path, 'w')

    # DEBUG mode!
    if args.debug_sample_size != None:
        DEBUG_SAMPLE_SIZE = args.debug_sample_size #300

        prompts = prompts[:DEBUG_SAMPLE_SIZE]
        gens = gens[:DEBUG_SAMPLE_SIZE]

    final_ner_score, final_contradict_prob, final_neutral_prob, final_entail_prob = 0, 0, 0, 0
    evaluation_cnt = 0

    total_nofact, total_checkworthy, total_offtopic = 0,0,0
    total_nofact_ratio, total_checkworthy_ratio, total_offtopic_ratio = 0,0,0

    all_nli_labels = []

    no_wiki_cnt = 0

    inccorect_ner_refute, correct_ner_nonrefute = 0, 0 # ideal cases
    correct_ner_bad_nli, incorrect_ner_good_nli = 0, 0 # unideal cases

    lm_nonfactual_analysis_list = []
    all_analysis_list = []

    for idx, (prompt_obj, gen_obj) in tqdm(enumerate(zip(prompts, gens)), total=len(prompts)):

        # assert prompt_obj['prompt'] == gen_obj['prompt']

        prompt_wiki_names = [ev_infos[0] for ev_infos in prompt_obj['evidence_info']]
        # prompt_wiki_names = [(ev_infos[0], 'GOLD') for ev_infos in prompt_obj['evidence_info']]
        # prompt_wiki_names = [wiki_name[0] for wiki_name in prompt_wiki_names]
        

        cnts_, metrics_ = single_instance_eval(gen_obj, prompt_wiki_names, run_nli_metric, run_ner_metric, args.test_og_fever)

        total_nofact += cnts_["NO_FACT"]
        total_checkworthy += cnts_["CHECK_WORTHY"]
        total_offtopic += cnts_["OFF_TOPIC"]

        if metrics_ != None:
            # print("\n[PROMPT {}] {}".format(idx, gen_obj['prompt']))
            # print("[LM continuation {}] {}".format(idx, gen_obj['text']))
            # print("[CHECKWORTHY {}] {}".format(idx, metrics_['claim_verified']))


            evaluation_cnt += 1 # important to keep this counter! Cuz, TEST_SAMPLE_SIZE != evaluation_cnt. only checkworthy claims are evaluated
            final_ner_score += metrics_['ner']
            final_contradict_prob += metrics_['nli-contr']
            final_neutral_prob += metrics_['nli-neutr']
            final_entail_prob += metrics_['nli-entail']
            all_nli_labels.append(metrics_['nli-label'])

            all_analysis_list.append(
                {
                    'prompt': gen_obj['prompt'],
                    'lm-gen': metrics_['claim_verified'],
                    'wiki': " ".join(prompt_wiki_names),
                    'ner': metrics_['ner'],
                    'nli-label': metrics_['nli-label'],
                    'nli-prob': metrics_['nli-contr']
                }
            )

            if run_ner_metric and run_nli_metric and args.save_gen_for_analysis:
                # do some analysis
                if metrics_['ner'] < 1.0 and metrics_['nli-label'] == 0:
                    # CASE: FIRMLY NONFACTUAL. if there is hallucinated NER AND NLI class is refute
                    inccorect_ner_refute += 1

                    lm_nonfactual_analysis_list.append({
                        'prompt': gen_obj['prompt'],
                        'lm-gen': metrics_['claim_verified'],
                        'wiki': " ".join(prompt_wiki_names),
                        'ner': metrics_['ner'],
                        'nli-label': metrics_['nli-label'],
                        'nli-prob': metrics_['nli-contr']
                    })

                elif metrics_['ner'] == 1.0 and metrics_['nli-label'] != 0:
                    # No hallucinated NER AND NLI class = support/NEI
                    correct_ner_nonrefute += 1

                elif metrics_['ner'] == 1.0 and metrics_['nli-label'] == 0:
                    # NER-NLI contradicting case 1: perfect NER score, but bad NLI 
                    correct_ner_bad_nli += 1

                elif metrics_['ner'] < 1.0 and metrics_['nli-label'] != 0:
                    # NER-NLI contradicting case 2: imperfect NER score, but not bad NLI 
                    incorrect_ner_good_nli += 1
    
    # analysis
    if args.save_gen_for_analysis:
        extra_name = args.gen_path.replace(".jsonl", "")

        df = pd.DataFrame(lm_nonfactual_analysis_list)
        df.to_csv("{}_badGen.csv".format(extra_name))

        df = pd.DataFrame(all_analysis_list)
        df.to_csv("{}_allGen.csv".format(extra_name))

    total_total = total_nofact + total_checkworthy + total_offtopic
    print("NO FACT: {:.2f}%, CHECKWORTHY: {:.2f}%, OFFTOPIC: {:.2f}%".format(total_nofact/total_total*100, total_checkworthy/total_total*100, total_offtopic/total_total*100))
    # print("NO FACT %: {}, CHECKWORTHY %: {}, OFFTOPIC %: {}".format(total_nofact_ratio/len(gens), total_checkworthy_ratio/len(gens), total_offtopic_ratio/len(gens)))
        
    print("NER: {:.2f}%".format(final_ner_score/evaluation_cnt*100))
    print("AVG PROBS: Contradict: {:.2f}%, Neutral: {:.2f}%, Entail: {:.2f}%".format(final_contradict_prob/evaluation_cnt*100, final_neutral_prob/evaluation_cnt*100, final_entail_prob/evaluation_cnt*100))

    if run_nli_metric:
        nli_counter = Counter(all_nli_labels)
        print("NLI CLASS %: Contradict: {:.2f}%, Neutral: {:.2f}%, Entail: {:.2f}%".format(
            nli_counter[0]/(nli_counter[0]+nli_counter[1]+nli_counter[2])*100,
            nli_counter[1]/(nli_counter[0]+nli_counter[1]+nli_counter[2])*100,
            nli_counter[2]/(nli_counter[0]+nli_counter[1]+nli_counter[2])*100
            ))

    if run_ner_metric and run_nli_metric and args.save_gen_for_analysis:
        print("IDEAL. inccorect_ner_refute: {}, correct_ner_nonrefute: {}".format(inccorect_ner_refute, correct_ner_nonrefute))
        print("NOT IDEAL. correct_ner_bad_nli: {}, incorrect_ner_good_nli: {}".format(correct_ner_bad_nli, incorrect_ner_good_nli))

    print("Evaluated {} samples!".format(evaluation_cnt))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--prompt_type', type=str, help='name of prompt type of the testset [factual, nonfactual]')
    parser.add_argument('--model_size', type=str, default='1.3b', help='LM model size')
    parser.add_argument('--exp_name', type=str, default='', help='experiment name for different generation techniques') 
    parser.add_argument('--debug_sample_size', type=int, default=None, help='# of sample size to use for debugging purpose. providing this value will automatically lead to debug mode')
    parser.add_argument('--gen_path', type=str, default=None, help='path to generations to evaluate') 

    parser.add_argument('--test_og_fever', action='store_true', help='Evaluate fever claims with label') 
    parser.add_argument('--save_gen_for_analysis', action='store_true', help='Flag for saving some lm-gens with its metric for analysis') 



    args = parser.parse_args()
    main(args)

    print("yay!")