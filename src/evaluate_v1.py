
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import re
import json

import spacy
# spacy.prefer_gpu()
spacy.load('en_core_web_sm')
nlp = spacy.load("en_core_web_sm")

import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from retriever import obtain_wiki, obtain_relevant_evidences
from metric import nli_metric, ner_metric

# DATA_DIR = '/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl'
# HOME_DIR = '/home/nayeonl/megatron-lm/'
from const import DATA_DIR, HOME_DIR

def evaluation_pipeline(obj, wiki_names, run_nli_metric, run_ner_metric, verbose=False):

    # 2. we need to further retrieve evidence for EACH sentences in the generated sentences -- mainly for those important NE
    # MUST: 'PERSON', 'ORG'
    # HMMM NOT SO SURE: 'LOC', GPE (Berlin, United States, US), 'NORP' (German, American, Russian, etc)
    # FULL LIST: ['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']


    prompt = obj['prompt']
    prompt_ne = obtain_important_ne(prompt, is_main_prompt=True) 
    # TODO use the Wikipedia list from the FEVER dataset!! Cuz there can be ambiguous cases --> issue: Savage has many wikipedia pages.

    text = obj['text']
    sents = sent_tokenize(text)
    sents_with_ne = [obtain_important_ne(sent, is_main_prompt=False) for sent in sents]


    wiki_names = prompt_ne['important_ne'] if wiki_names == [] else wiki_names
    wiki_names = [wiki_name[0] for wiki_name in wiki_names]


    visited_wiki_names = set(wiki_names)

    wiki_sentences = obtain_wiki(wiki_names) # init wiki_sentence list with wiki-about-prompt
    
    nli_contradict_prob, nli_entail_prob, nli_neutral_prob = 0, 0, 0
    correct_ner_ratio = 0
    cnt = 0

    for claim_obj in sents_with_ne:

        if len(claim_obj['important_ne']) + len(claim_obj['unimportant_ne']) > 0:
            cnt+=1
            claim_to_verify = claim_obj['sent']
            
            claim_specific_ne = [ne[0] for ne in claim_obj['important_ne'] if ne[0] not in visited_wiki_names]
            # obtain more wiki sents for new important NEs that appear in the claim
            if len(claim_specific_ne) > 0:
                extra_wiki_candidates = obtain_wiki(claim_specific_ne)

                '''
                    Combine prompt-topic wiki sents with claim specific topic wiki sents
                    
                    Note: we are gradually incrementing the pool of wiki_candidates, as we see more LM continuations
                    Since LM continuations are dependent on the previous context, it is possible that the continuation contains 
                    information about NE that DOES NOT appear in the original prompt -- instead NE that appeared in one of the earlier continuation.
                '''
                wiki_sentences.extend(extra_wiki_candidates)

                visited_wiki_names.update(set(claim_specific_ne))

            if wiki_sentences == []:
                print("ATTENTION!! EMPTY WIKI!! FOR CLAIM: ", wiki_names, claim_to_verify)
                continue

            topk_evs = obtain_relevant_evidences(claim_to_verify, wiki_sentences, k=3, method='tfidf')

            if run_ner_metric:
                _correct_ner_ratio = ner_metric(claim_obj['important_ne'] + claim_obj['unimportant_ne'], wiki_sentences) # apply directly on wiki
                correct_ner_ratio += _correct_ner_ratio

            if run_nli_metric:
                # TODO for now im using top-1 evidence, but i think it is more robust to use top-2. but how? 
                top1_evidence = topk_evs[0][0]
                nli_scores = nli_metric(claim_to_verify, top1_evidence) # [[contradiction, neutral, entailment], argmax]
                nli_contradict_prob += nli_scores[0][0][0] 
                nli_neutral_prob += nli_scores[0][0][1] 
                nli_entail_prob += nli_scores[0][0][2] 

                # TODO: Question = what about neutral???? do we penalize? or not? check other literature

    cnt = 1 if cnt == 0 else cnt # to avoid division-by-zero error
    if run_ner_metric:
        correct_ner_ratio = correct_ner_ratio/cnt
        if verbose:
            print("AVG NER ``Precision'' (): {:.4f}".format(correct_ner_ratio))

    if run_nli_metric:
        nli_contradict_prob = nli_contradict_prob/cnt
        nli_neutral_prob = nli_neutral_prob/cnt
        nli_entail_prob = nli_entail_prob/cnt

        if verbose:
            print("AVG contradict: {:.4f}, neutral: {:.4f}, entail: {:.4f}".format(
                nli_contradict_prob,
                nli_neutral_prob,
                nli_entail_prob
            ))
    
    return correct_ner_ratio, nli_contradict_prob, nli_neutral_prob, nli_entail_prob


IMPORTANT_ENT_TYPE = set(['ORG', 'PERSON', 'WORK_OF_ART', 'PRODUCT', 'EVENT'])
def obtain_important_ne(sent, is_main_prompt):
    important_words = []
    
    doc = nlp(sent)
    ents = [(ent.text, ent.label_) for ent in doc.ents]

    if is_main_prompt:
        capitalized_words = re.findall('([A-Z][a-z]+)', sent)
        # capitalized_words = re.findall('(?<!^)([A-Z][a-z]+)( ([A-Z][a-z]+))*', sent)

        # filter any stopword or those that already appear in ents list
        ents_texts = " ".join([ent[0] for ent in ents])
        capitalized_words = [(word, 'CAPITALIZED') for word in capitalized_words if word not in ents_texts and word.lower() not in stop_words]
        important_words.extend(capitalized_words)

    important_words.extend([ent for ent in ents if ent[1] in IMPORTANT_ENT_TYPE])
    remaining_ne = [ent for ent in ents if ent[1] not in IMPORTANT_ENT_TYPE]

    sents_with_ne = {
                        "sent": sent,
                        "important_ne": important_words,
                        "unimportant_ne": remaining_ne
                    }

    return sents_with_ne 


def main():

    model_size = '1.3b'
    # prompt_type = 'nonfactual' # 1500
    prompt_type = 'factual' 


    prompt_path = '{}/prompts/fever_dev_{}.jsonl'.format(HOME_DIR, prompt_type)
    gen_path = '{}/generations/{}.{}_fever.jsonl'.format(DATA_DIR, model_size, prompt_type)
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

    # DEBUG PURPOSE
    SAMPLE_SIZE = 1500
    prompts = prompts[:SAMPLE_SIZE]
    gens = gens[:SAMPLE_SIZE]

    final_ner_score, final_contradict_prob, final_neutral_prob, final_entail_prob = 0, 0, 0, 0
    
    for prompt_obj, gen_obj in tqdm(zip(prompts, gens), total=len(prompts)):

        assert prompt_obj['prompt'] == gen_obj['prompt']

        prompt_wiki_names = [(ev_infos[0], 'GOLD') for ev_infos in prompt_obj['evidence_info']]
        
        # print("LM continuation: ", gen_obj['text'])
        correct_ner_ratio, nli_contradict_prob, nli_neutral_prob, nli_entail_prob = evaluation_pipeline(gen_obj, prompt_wiki_names, run_nli_metric=True, run_ner_metric=False)

        final_ner_score += correct_ner_ratio
        final_contradict_prob += nli_contradict_prob
        final_neutral_prob += nli_neutral_prob
        final_entail_prob += nli_entail_prob

        outfile.write('{}, {}, {}, {}\n'.format(correct_ner_ratio, nli_contradict_prob, nli_neutral_prob, nli_entail_prob))
    
    print("NER: {}".format(final_ner_score/SAMPLE_SIZE))
    print("Contradict: {}, Neutral: {}, Entail: {}".format(final_contradict_prob/SAMPLE_SIZE, final_neutral_prob/SAMPLE_SIZE, final_entail_prob/SAMPLE_SIZE))

    outfile.close()
    


if __name__ == '__main__':

    main()

    print("yay!")