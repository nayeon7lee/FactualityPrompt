from mailbox import linesep
import requests
import json
import collections
import bs4
from nltk.tokenize import sent_tokenize

from os import listdir
from os.path import isfile, join

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import fuzz, process

# import datasets
# import bert_score
from sentence_transformers import SentenceTransformer, util

from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
# from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

from src.const import DATA_DIR, HOME_DIR
# from fever_athene.src.retrieval.fever_doc_db import FeverDocDB

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def _fetchUrl(url):
    headers = {
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36',
    }

    r = requests.get(url, headers=headers)
    r.raise_for_status()
    r.encoding = r.apparent_encoding
    return r.text

def _text_process(text):
    text = text.replace("\n", " ")
    # TODO remove [##]. e.g., [2],[3]
    return text


'''
    Returns wiki sentences for ALL entities that appear in the provided prompt_ents
'''
def obtain_wiki_from_internet(prompt_ents, offline=False):
    all_wiki_sents = []

    error_log_path = '{}/wiki_error.log'.format(HOME_DIR)
    log_file = open(error_log_path, 'a')

    # 0. map entities to wiki-doc-name
    for ent in prompt_ents:

        # try:
        wiki_doc_name = ent.strip().replace(" ", "%20") # 'Telemundo'
        # print("Retrieving {}...".format(wiki_doc_name))

        # 1. get wiki-obj
        wiki_dir = '{}/wiki'.format(DATA_DIR)
        wiki_fnames = [f for f in listdir(wiki_dir)]

        f_name = wiki_doc_name + '.json'
        if f_name in wiki_fnames:
            # load 
            with open("{}/{}".format(wiki_dir, f_name), 'r') as infile:
                wiki_obj = json.load(infile)

        else: # wiki doesnt exist in our file system

            if offline: # in offline mode, we treat this as "evidence not found"
                return []
                
            else: # in online mode, we try to find the wiki from Wikipedia. 
                
                

                wiki_url = "https://en.wikipedia.org/w/api.php?action=parse&page={}&prop=text&format=json".format(wiki_doc_name)

                html = _fetchUrl(wiki_url)
                bsobj = bs4.BeautifulSoup(html,'html.parser')

                wiki_obj = collections.defaultdict(list)

                ## parse the main
                wiki_section_name = 'Main'

                for obj in bsobj.find_all(['p', 'h2', 'h3']):
                    if obj.name == 'h2':
                        wiki_section_name = obj.get_text()
                    elif obj.name == 'h3':
                        text = _text_process("<== " + obj.get_text() + " ==>")
                        
                        wiki_obj[wiki_section_name].append(text)
                    else:
                        text = _text_process(obj.get_text())
                        wiki_obj[wiki_section_name].append(text)

                        if "may refer to:" in text or "Redirect to:" in text:
                            # IR fail case: this wikipedia is ambiguation resolving wiki so return []
                            # -- e.g., https://en.wikipedia.org/wiki/Celtic
                            log_file.write("{} /// {}\n".format(wiki_doc_name, "REDIRECT_TO"))
                            return []

                
                if len(wiki_obj.keys()) == 0: # IR fail case: wiki does not exist. empty page
                    # print("Failed wiki name: {}".format(wiki_doc_name))
                    log_file.write("{} /// {}\n".format(wiki_doc_name, "NO_WIKI"))
                    return []

            
                # parse the table
                infobox_obj = bsobj.find("table")

                #  TODO infobox parsing can be improved... 
                # e.g., cannot handle tables that exist in "Horse".

                if infobox_obj != None:
                    infobox_dict = collections.defaultdict(list)
                    current_header = 'Main'
                    current_label = ''
                    
                    try:
                        if type(infobox_obj.contents[0])==bs4.element.NavigableString:
                            infobox_contents = infobox_obj.contents[1]
                        else:
                            infobox_contents = infobox_obj.contents[0]

                        for t_obj in infobox_contents.find_all(["td", "th"]):
                            try: 
                                class_name = str(t_obj['class'])
                            
                                if "infobox-header" in class_name:
                                    current_header = t_obj.get_text()

                                elif "infobox-label" in class_name:
                                    current_label = t_obj.get_text()

                                elif "infobox-data" in class_name:
                                    current_data = t_obj.get_text()
                                    infobox_dict[current_header].append((current_label, current_data))
                            except:
                                # print("ERROR:", t_obj)
                                continue
                    except:
                        continue

                    wiki_obj['Infobox'].append(infobox_dict)
                else:
                    log_file.write("{} /// {}\n".format(wiki_doc_name, "NO_INFOBOX_TABLE"))


                with open("{}/{}".format(wiki_dir, f_name), 'w') as outfile:
                    json.dump(wiki_obj, outfile)

        # 2. transform wiki-doc to wiki-sentences
        wiki_sents = []
        for key in wiki_obj:
            if key == 'Infobox':
                # handle infobox info
                for info_header, info_label2data in wiki_obj['Infobox'][0].items():
                    for label_data_tuple in info_label2data:
                        kb_to_text = wiki_doc_name + " " + info_header + " " + label_data_tuple[0] + " " + label_data_tuple[1]
                        # print(kb_to_text)
                        wiki_sents.append(kb_to_text)
            else:

                wiki_content = " ".join(wiki_obj[key]).replace("\\n", " ")
                # concat name of the wiki-doc to all the wiki-sentences. later, we combine mupltiple wiki-docs together, 
                # and this trick helps us to ensure that we have some information linking the sent back to its origin
                wiki_doc_sents = [wiki_doc_name + " /// " + sent for sent in sent_tokenize(wiki_content)]
                wiki_sents.extend(wiki_doc_sents)

        all_wiki_sents.extend(wiki_sents)
        # except:
        #     log_file.write("{} /// {}\n".format(wiki_doc_name, "UNKNOWN_ERROR"))
        #     continue

    log_file.close()
    # print("Retrieved {}...".format(wiki_doc_name))
    return all_wiki_sents



def clean_wiki_sents(wiki_sents):
    # filter really short sentences --> things like headers
    filtered_wiki_sents = []
    for sent in wiki_sents:
        if len(sent.split(" ")) <= 4 or "Section::::" in sent:
            continue

        filtered_wiki_sents.append(sent)
    return filtered_wiki_sents


MODEL = SentenceTransformer('all-MiniLM-L6-v2')
def obtain_relevant_evidences(claim, wiki_sents, k, method):

    evs = []

    wiki_sents = clean_wiki_sents(wiki_sents)

    if method == 'tfidf' or method == 'combined':
        vectorizer = TfidfVectorizer()
        wiki_candidate_vectors = vectorizer.fit_transform(wiki_sents)
        
        query_vector = vectorizer.transform([claim])
        relevance_scores = cosine_similarity(wiki_candidate_vectors, query_vector).reshape((-1,)).tolist()

        evs.extend(sorted(zip(wiki_sents, relevance_scores), key=lambda x: -x[1])[:k])
        
    if method == 'emb_sim' or method == 'combined':
        ev_embeddings = MODEL.encode(wiki_sents)
        q_embedding = MODEL.encode(claim)

        hits = util.semantic_search(q_embedding, ev_embeddings, top_k=k)
        # print("[SENT_EMB]")
        # print([(wiki_sents[int(id_dict['corpus_id'])], id_dict['score']) for id_dict in hits[0]])
        # print("\n")

        evs.extend([(wiki_sents[int(id_dict['corpus_id'])], id_dict['score']) for id_dict in hits[0]])

    return evs

def obtain_relevant_infobox(claim, infobox_candidates, k):

    k_infobox_items = process.extract(claim, infobox_candidates, limit=2)

    return k_infobox_items

DB = FeverDocDB(path = "{}/data/kilt_db.db".format(HOME_DIR))
def get_wiki_from_db(wiki_names):
    
    all_lines = []
    for wiki_name in wiki_names:
        
        lines = DB.get_doc_lines(wiki_name)
        if lines != None:
            all_lines.extend(sent_tokenize(lines))
            
    return all_lines

if __name__ == '__main__':

    MODEL = SentenceTransformer('all-MiniLM-L6-v2')

    print("yay!")
