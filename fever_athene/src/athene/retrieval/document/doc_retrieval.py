import argparse
import json
import os
import re
from multiprocessing.pool import ThreadPool

import nltk
import wikipedia
from allennlp.service.predictors import Predictor
from drqa.retriever.utils import normalize
from tqdm import tqdm

from common.dataset.reader import JSONLineReader
from retrieval.fever_doc_db import FeverDocDB


def processed_line(method, line):
    nps, wiki_results, pages = method.exact_match(line)
    line['noun_phrases'] = nps
    line['predicted_pages'] = pages
    line['wiki_results'] = wiki_results
    return line


# def wiki_search(method,line,k_first_pages=None,add_claim=False):
#     wiki_results = method.get_doc_for_claim(line,k_first_pages=k_first_pages)
#     line['wiki_results'] = list(set(wiki_results))
#     return line


class Doc_Retrieval():

    def __init__(self, database_path, add_claim=False, k_wiki_results=None):
        self.db = FeverDocDB(database_path)
        self.add_claim = add_claim
        self.k_wiki_results = k_wiki_results
        self.proter_stemm = nltk.PorterStemmer()
        self.tokenizer = nltk.word_tokenize
        self.predictor = Predictor.from_path(
            "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")

    def get_NP(self, tree, nps):

        if isinstance(tree, dict):
            if "children" not in tree:
                if tree['nodeType'] == "NP":
                    # print(tree['word'])
                    # print(tree)
                    nps.append(tree['word'])
            elif "children" in tree:
                if tree['nodeType'] == "NP":
                    # print(tree['word'])
                    nps.append(tree['word'])
                    self.get_NP(tree['children'], nps)
                else:
                    self.get_NP(tree['children'], nps)
        elif isinstance(tree, list):
            for sub_tree in tree:
                self.get_NP(sub_tree, nps)

        return nps

    def get_subjects(self, tree):
        subject_words = []
        subjects = []
        for subtree in tree['children']:
            if subtree['nodeType'] == "VP" or subtree['nodeType'] == 'S' or subtree['nodeType'] == 'VBZ':
                subjects.append(' '.join(subject_words))
                subject_words.append(subtree['word'])
            else:
                subject_words.append(subtree['word'])
        return subjects

    def get_noun_phrases(self, line):

        claim = line['claim']
        tokens = self.predictor.predict(claim)
        nps = []
        tree = tokens['hierplane_tree']['root']
        noun_phrases = self.get_NP(tree, nps)
        subjects = self.get_subjects(tree)
        for subject in subjects:
            if len(subject) > 0:
                noun_phrases.append(subject)
        if self.add_claim:
            noun_phrases.append(claim)
        return list(set(noun_phrases))

    def get_doc_for_claim(self, noun_phrases):

        predicted_pages = []
        for np in noun_phrases:
            if len(np) > 300:
                continue
            docs = wikipedia.search(np)
            if self.k_wiki_results is not None:
                predicted_pages.extend(docs[:self.k_wiki_results])
            else:
                predicted_pages.extend(docs)
            # sleep_num = random.uniform(0.1,0.7)
            # time.sleep(sleep_num)
        predicted_pages = set(predicted_pages)
        processed_pages = []
        for page in predicted_pages:
            page = page.replace(" ", "_")
            page = page.replace("(", "-LRB-")
            page = page.replace(")", "-RRB-")
            page = page.replace(":", "-COLON-")
            processed_pages.append(page)

        return processed_pages

    def np_conc(self, noun_phrases):

        noun_phrases = set(noun_phrases)
        predicted_pages = []
        for np in noun_phrases:
            page = np.replace('( ', '-LRB-')
            page = page.replace(' )', '-RRB-')
            page = page.replace(' - ', '-')
            page = page.replace(' -', '-')
            page = page.replace(' :', '-COLON-')
            page = page.replace(' ,', ',')
            page = page.replace(" 's", "'s")
            page = page.replace(' ', '_')

            if len(page) < 1:
                continue
            doc_lines = self.db.get_doc_lines(page)
            if doc_lines is not None:
                predicted_pages.append(page)
        return predicted_pages

    def exact_match(self, line):

        noun_phrases = self.get_noun_phrases(line)
        wiki_results = self.get_doc_for_claim(noun_phrases)
        wiki_results = list(set(wiki_results))

        claim = normalize(line['claim'])
        claim = claim.replace(".", "")
        claim = claim.replace("-", " ")
        words = [self.proter_stemm.stem(word.lower()) for word in self.tokenizer(claim)]
        words = set(words)
        predicted_pages = self.np_conc(noun_phrases)

        for page in wiki_results:
            page = normalize(page)
            processed_page = re.sub("-LRB-.*?-RRB-", "", page)
            processed_page = re.sub("_", " ", processed_page)
            processed_page = re.sub("-COLON-", ":", processed_page)
            processed_page = processed_page.replace("-", " ")
            processed_page = processed_page.replace("–", " ")
            processed_page = processed_page.replace(".", "")
            page_words = [self.proter_stemm.stem(word.lower()) for word in self.tokenizer(processed_page) if
                          len(word) > 0]

            if all([item in words for item in page_words]):
                if ':' in page:
                    page = page.replace(":", "-COLON-")
                predicted_pages.append(normalize(page))
        predicted_pages = list(set(predicted_pages))
        # print("claim: ",claim)
        # print("nps: ",noun_phrases)
        # print("wiki_results: ",wiki_results)
        # print("predicted_pages: ",predicted_pages)
        # print("evidence:",line['evidence'])
        return noun_phrases, wiki_results, predicted_pages


def get_map_function(parallel):
    return p.imap_unordered if parallel else map


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--db-file', type=str, help="database file which saves pages")
    parser.add_argument('--in-file', type=str, help="input dataset")
    parser.add_argument('--out-file', type=str, help="path to save output dataset")
    parser.add_argument('--k-wiki', type=int, help="first k pages for wiki search")
    parser.add_argument('--parallel', type=bool, default=True)
    parser.add_argument('--add-claim', type=bool, default=True)
    args = parser.parse_args()

    # tfidf_path = "data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz"
    method = Doc_Retrieval(database_path=args.db_file, add_claim=args.add_claim, k_wiki_results=args.k_wiki)
    processed = dict()

    path = os.getcwd()
    with open(os.path.join(path, args.in_file), "r") as f, open(os.path.join(path, args.out_file), "w+") as f2:
        jlr = JSONLineReader()
        lines = jlr.process(f)

        with ThreadPool(processes=4) as p:
            for line in tqdm(get_map_function(args.parallel)(lambda line: processed_line(method, line), lines),
                             total=len(lines)):
                processed[line['id']] = line
                # time.sleep(0.5)

        for line in lines:
            f2.write(json.dumps(processed[line['id']]) + "\n")
