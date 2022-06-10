import os
import pickle
import random
from multiprocessing.pool import ThreadPool

from pyfasttext import FastText

import nltk
import numpy as np
from tqdm import tqdm

from common.dataset.reader import JSONLineReader
from retrieval.fever_doc_db import FeverDocDB

def processed_line(method, line):
    nps, wiki_results, pages = method.exact_match(line)
    line['noun_phrases'] = nps
    line['predicted_pages'] = pages
    line['wiki_results'] = wiki_results
    return line



class Data(object):

    def __init__(self,  embedding_path, train_file, dev_file, test_file, fasttext_path, num_negatives, h_max_length,
                 s_max_length, random_seed, reserve_embed=False, db_filepath="data/fever/fever.db", load_instances=True, retrieval=None):

        self.random_seed = random_seed
        self.retrieval = retrieval
        self.embedding_path = embedding_path
        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file
        self.fasttext_path = fasttext_path
        self.num_negatives = num_negatives
        self.h_max_length = h_max_length
        self.s_max_length = s_max_length
        self.db_filepath = db_filepath
        self.db = FeverDocDB(self.db_filepath)
        self.reserve_embed = reserve_embed
        self.load_instances = load_instances

        self.data_pipeline()

    def data_pipeline(self):

        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # create diretory to store sampling data and processed data
        # store_dir = "data.h{}.s{}.seed{}".format(self.h_max_length, self.s_max_length, self.random_seed)
        # self.absou_dir = os.path.join(base_dir, store_dir)
        os.makedirs(self.embedding_path, exist_ok=True)

        print("getting training data pages")
        train_data_path = os.path.join(self.embedding_path, "train_sample.p")
        X_train = self.train_data_loader(train_data_path, self.train_file, num_samples=self.num_negatives)
        dev_datapath = os.path.join(self.embedding_path, "dev_data.p")

        print("getting dev data pages")
        devs, self.dev_labels = self.dev_data_loader(dev_datapath, self.dev_file)
        if self.test_file is None:
            self.test_file = self.dev_file


        print("getting test data pages")
        test_datapath = os.path.join(self.embedding_path, "test_data.p")
        tests, self.test_location_indexes = self.predict_data_loader(test_datapath, self.test_file)

        words_dict_path = os.path.join(self.embedding_path, "words_dict.p")
        if os.path.exists(words_dict_path):
            with open(words_dict_path, "rb") as f:
                self.word_dict = pickle.load(f)
        else:
            self.word_dict = self.get_complete_words(words_dict_path, X_train, devs, tests)

        self.iword_dict = self.inverse_word_dict(self.word_dict)

        if self.load_instances:
            train_indexes_path = os.path.join(self.embedding_path, "train_indexes.p")
            self.X_train_indexes = self.train_indexes_loader(train_indexes_path, X_train)
            dev_indexes_path = os.path.join(self.embedding_path, "dev_indexes.p")
            self.dev_indexes = self.predict_indexes_loader(dev_indexes_path, devs)
            test_indexes_path = os.path.join(self.embedding_path, "test_indexes.p")
            self.test_indexes = self.predict_indexes_loader(test_indexes_path, tests)

            embed_dict = self.load_fasttext(self.iword_dict)
            print("embed_dict size {}".format(len(embed_dict)))
            _PAD_ = len(self.word_dict)
            self.word_dict[_PAD_] = '[PAD]'
            self.iword_dict['[PAD]'] = _PAD_
            self.embed = self.embed_to_numpy(embed_dict)

        return self

    def get_whole_evidence(self, evidence_set, db):
        pos_sents = []
        for evidence in evidence_set:
            page = evidence[2]
            doc_lines = db.get_doc_lines(page)
            doc_lines = self.get_valid_texts(doc_lines, page)
            for doc_line in doc_lines:
                if doc_line[2] == evidence[3]:
                    pos_sents.append(doc_line[0])
        pos_sent = ' '.join(pos_sents)
        return pos_sent

    def get_valid_texts(self, lines, page):
        if not lines:
            return []
        doc_lines = [doc_line.split("\t")[1] if len(doc_line.split("\t")[1]) > 1 else "" for doc_line in
                     lines.split("\n")]
        doc_lines = list(zip(doc_lines, [page] * len(doc_lines), range(len(doc_lines))))
        return doc_lines

    def handle(self,line,num_sample):
        ret = []
        if not "predicted_pages" in line:
            line = processed_line(self.retrieval, line)
        pos_pairs = []
        # count1 += 1
        if line['label'].upper() == "NOT ENOUGH INFO":
            return []
        neg_sents = []
        claim = line['claim']

        pos_set = set()
        for evidence_set in line['evidence']:
            pos_sent = self.get_whole_evidence(evidence_set, self.db)
            if pos_sent in pos_set:
                continue
            pos_set.add(pos_sent)

        p_lines = []
        evidence_set = set(
            [(evidence[2], evidence[3]) for evidences in line['evidence'] for evidence in evidences])

        pages = [page for page in line['predicted_pages'] if page is not None]

        for page in pages:
            doc_lines = self.db.get_doc_lines(page)
            p_lines.extend(self.get_valid_texts(doc_lines, page))
        for doc_line in p_lines:
            if (doc_line[1], doc_line[2]) not in evidence_set:
                neg_sents.append(doc_line[0])

        num_sampling = num_sample
        if len(neg_sents) < num_sampling:
            num_sampling = len(neg_sents)
            # print(neg_sents)
        if num_sampling == 0:
            return []
        else:
            for pos_sent in pos_set:
                samples = random.sample(neg_sents, num_sampling)
                for sample in samples:
                    if not sample:
                        continue
                    ret.append((claim, pos_sent, sample))
                    # if count % 1000 == 0:
                    #     print("claim:{} ,evidence :{} sample:{}".format(claim, pos_sent, sample))
        return ret
    def sampling(self, datapath, num_sample=1):

        jlr = JSONLineReader()
        ret = []
        print("sampling for "+ datapath)
        with open(datapath, "r") as f:
            lines = jlr.process(f)
            print(len(lines))
            with ThreadPool(processes=48) as p:
                for line in tqdm(p.imap(lambda x: self.handle(x, num_sample), lines),total=len(lines)):
                    if line is not None:
                        ret.extend(line)

        print("Done")

        return ret

    def predict_processing(self, datapath):

        jlr = JSONLineReader()

        devs = []
        all_indexes = []

        with open(datapath, "rb") as f:
            lines = jlr.process(f)

            for line in tqdm(lines):
                dev = []
                indexes = []
                pages = set()
                # pages = line['predicted_pages']
                pages.update(page for page in line['predicted_pages'])
                # if len(pages) == 0:
                #     pages.add("Michael_Hutchence")
                claim = line['claim']
                p_lines = []
                for page in pages:
                    doc_lines = self.db.get_doc_lines(page)
                    if not doc_lines:
                        continue
                    p_lines.extend(self.get_valid_texts(doc_lines, page))

                for doc_line in p_lines:
                    if not doc_line[0]:
                        continue
                    dev.append((claim, doc_line[0]))
                    indexes.append((doc_line[1], doc_line[2]))
                # print(len(dev))
                if len(dev) == 0:
                    dev.append((claim, 'no evidence for this claim'))
                    indexes.append(('empty', 0))
                devs.append(dev)
                all_indexes.append(indexes)
        return devs, all_indexes

    def dev_processing(self, data_path):

        jlr = JSONLineReader()

        with open(data_path, "r") as f:
            lines = jlr.process(f)

            devs = []
            labels = []
            for line in tqdm(lines):

                dev = []
                label = []
                if line['label'].upper() == "NOT ENOUGH INFO":
                    continue
                evidence_set = set(
                    [(evidence[2], evidence[3]) for evidences in line['evidence'] for evidence in evidences])

                pages = [page for page in line['predicted_pages'] if page is not None]
                for page, num in evidence_set:
                    pages.append(page)
                pages = set(pages)

                p_lines = []
                for page in pages:
                    doc_lines = self.db.get_doc_lines(page)
                    p_lines.extend(self.get_valid_texts(doc_lines, page))
                for doc_line in p_lines:
                    if not doc_line[0]:
                        continue
                    dev.append((line['claim'], doc_line[0]))
                    if (doc_line[1], doc_line[2]) in evidence_set:
                        label.append(1)
                    else:
                        label.append(0)
                if len(dev) == 0 or len(label) == 0:
                    continue
                devs.append(dev)
                labels.append(label)
        return devs, labels

    def train_data_loader(self, train_sampled_path, data_path, num_samples=1):

        if os.path.exists(train_sampled_path):
            with open(train_sampled_path, 'rb') as f:
                X = pickle.load(f)
        else:
            X = self.sampling(data_path, num_samples)
            with open(train_sampled_path, 'wb') as f:
                pickle.dump(X, f)
        return X

    def dev_data_loader(self, dev_data_path, data_path):

        if os.path.exists(dev_data_path):
            with open(dev_data_path, "rb") as f:
                data = pickle.load(f)
                devs, labels = zip(*data)
        else:
            devs, labels = self.dev_processing(data_path)
            data = list(zip(devs, labels))
            with open(dev_data_path, 'wb') as f:
                pickle.dump(data, f)
        return devs, labels

    def predict_data_loader(self, predict_data_path, data_path):

        if os.path.exists(predict_data_path):
            print(predict_data_path)
            with open(predict_data_path, "rb") as f:
                data = pickle.load(f)
                devs, location_indexes = zip(*data)
        else:
            devs, location_indexes = self.predict_processing(data_path)
            data = list(zip(devs, location_indexes))
            with open(predict_data_path, 'wb') as f:
                pickle.dump(data, f)
        return devs, location_indexes

    def sent_processing(self, sent):
        sent = sent.replace('\n', '')
        sent = sent.replace('-', ' ')
        sent = sent.replace('/', ' ')
        return sent

    def nltk_tokenizer(self, sent):
        # sent = sent_processing(sent)
        return nltk.word_tokenize(sent)

    def get_words(self, claims, sents):

        words = set()
        for claim in claims:
            for idx, word in enumerate(self.nltk_tokenizer(claim)):
                if idx >= self.h_max_length:
                    break
                words.add(word.lower())
        for sent in sents:
            for idx, word in enumerate(self.nltk_tokenizer(sent)):
                if idx >= self.s_max_length:
                    break
                words.add(word.lower())
        return words

    def get_train_words(self, X):
        claims = set()
        sents = []
        for claim, pos, neg in X:
            claims.add(claim)
            sents.append(pos)
            sents.append(neg)

        train_words = self.get_words(claims, sents)
        print("training words processing done!")
        return train_words

    def get_predict_words(self, devs):
        dev_words = set()
        # nlp = StanfordCoreNLP(corenlp_path)
        for dev in tqdm(devs):
            claims = set()
            sents = []
            for pair in dev:
                claims.add(pair[0])
                sents.append(pair[1])
            dev_tokens = self.get_words(claims, sents)
            dev_words.update(dev_tokens)
        print("dev_words processing done!")
        return dev_words

    def word_2_dict(self, words):
        word_dict = {}
        for idx, word in enumerate(words):
            word = word.replace('\n', '')
            word = word.replace('\t', '')
            word_dict[idx] = word

        return word_dict

    def inverse_word_dict(self, word_dict):

        iword_dict = {}
        for key, word in word_dict.items():
            iword_dict[word] = key
        return iword_dict

    def load_fasttext(self, iword_dict):

        embed_dict = {}
        print(self.fasttext_path)
        model = FastText(self.fasttext_path)
        for word, key in iword_dict.items():
            embed_dict[key] = model[word]
            # print(embed_dict[key])
        print('Embedding size: %d' % (len(embed_dict)))
        return embed_dict

    def embed_to_numpy(self, embed_dict):

        feat_size = len(embed_dict[list(embed_dict.keys())[0]])
        if self.reserve_embed:
            embed = np.zeros((len(embed_dict) + 200000 + 1, feat_size), np.float32)
        else:
            embed = np.zeros((len(embed_dict) + 1, feat_size), np.float32)
        for k in embed_dict:
            embed[k] = np.asarray(embed_dict[k])
        print('Generate numpy embed:', embed.shape)

        return embed

    def sent_2_index(self, sent, word_dict, max_length):
        words = self.nltk_tokenizer(sent)
        word_indexes = []
        for idx, word in enumerate(words):
            if idx >= max_length:
                break
            else:
                word_indexes.append(word_dict[word.lower()])
        return word_indexes

    def train_data_indexes(self, X, word_dict):

        X_indexes = []
        print("start index words into integers")
        for claim, pos, neg in X:
            claim_indexes = self.sent_2_index(claim, word_dict, self.h_max_length)
            pos_indexes = self.sent_2_index(pos, word_dict, self.s_max_length)
            neg_indexes = self.sent_2_index(neg, word_dict, self.s_max_length)
            X_indexes.append((claim_indexes, pos_indexes, neg_indexes))
        print('Training data size:', len(X_indexes))
        return X_indexes

    def predict_data_indexes(self, data, word_dict):

        devs_indexes = []
        for dev in data:
            sent_indexes = []
            claim = dev[0][0]
            claim_index = self.sent_2_index(claim, word_dict, self.h_max_length)
            claim_indexes = [claim_index] * len(dev)
            for claim, sent in dev:
                sent_index = self.sent_2_index(sent, word_dict, self.s_max_length)
                sent_indexes.append(sent_index)
            assert len(sent_indexes) == len(claim_indexes)
            dev_indexes = list(zip(claim_indexes, sent_indexes))
            devs_indexes.append(dev_indexes)
        return devs_indexes

    def get_complete_words(self, words_dict_path, train_data, dev_data, test_data):
        print("getting complete words")

        all_words = set()
        train_words = self.get_train_words(train_data)
        all_words.update(train_words)
        dev_words = self.get_predict_words(dev_data)
        all_words.update(dev_words)
        test_words = self.get_predict_words(test_data)
        all_words.update(test_words)
        word_dict = self.word_2_dict(all_words)
        with open(words_dict_path, "wb") as f:
            pickle.dump(word_dict, f)

        return word_dict

    def train_indexes_loader(self, train_indexes_path, train_data):

        if os.path.exists(train_indexes_path):
            with open(train_indexes_path, "rb") as f:
                X_indexes = pickle.load(f)
        else:
            X_indexes = self.train_data_indexes(train_data, self.iword_dict)
            with open(train_indexes_path, "wb") as f:
                pickle.dump(X_indexes, f)
        return X_indexes

    def predict_indexes_loader(self, predict_indexes_path, predict_data):

        if os.path.exists(predict_indexes_path):
            with open(predict_indexes_path, "rb") as f:
                predicts_indexes = pickle.load(f)
        else:
            predicts_indexes = self.predict_data_indexes(predict_data, self.iword_dict)
            with open(predict_indexes_path, "wb") as f:
                pickle.dump(predicts_indexes, f)
        return predicts_indexes

    def update_word_dict(self, test_path):

        self.new_test_datapath = os.path.join(self.embedding_path, "new_test_data.p")
        new_tests, self.test_location_indexes = self.predict_data_loader(self.new_test_datapath, test_path)

        new_test_words = self.get_predict_words(new_tests)
        print(len(self.iword_dict))
        print(len(self.word_dict))
        self.test_words_dict = {}
        for word in new_test_words:
            if word not in self.iword_dict:
                idx = len(self.word_dict)
                self.word_dict[idx] = word
                self.test_words_dict[idx] = word

        self.iword_dict = self.inverse_word_dict(self.word_dict)
        self.test_iword_dict = self.inverse_word_dict(self.test_words_dict)

        print("updated iword dict size: ", len(self.iword_dict))
        print("test iword dict size: ", len(self.test_iword_dict))

    def update_embeddings(self):

        test_embed_dict = self.load_fasttext(self.test_iword_dict)

        for k in test_embed_dict:
            self.embed[k] = np.asarray(test_embed_dict[k])
        print("updated embed size: ", self.embed.shape)

    def get_new_test_indexes(self, test_path):

        new_tests, self.new_test_location_indexes = self.predict_data_loader(self.new_test_datapath, test_path)

        new_tests_indexes_path = os.path.join(self.embedding_path, "new_test_indexes.p")
        self.new_tests_indexes = self.predict_indexes_loader(new_tests_indexes_path, new_tests)
