import os
import pickle
import random

import nltk
import numpy as np
from tqdm import tqdm

from common.dataset.reader import JSONLineReader
from retrieval.fever_doc_db import FeverDocDB


class ELMO_Data(object):

    def __init__(self, base_path, train_file, dev_file, test_file, num_negatives, h_max_length, s_max_length,
                 random_seed=100, db_filepath="data/fever/fever.db"):

        self.random_seed = random_seed

        self.base_path = base_path
        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file
        self.num_negatives = num_negatives
        self.h_max_length = h_max_length
        self.s_max_length = s_max_length
        self.db_filepath = db_filepath
        self.db = FeverDocDB(self.db_filepath)

        self.data_pipeline()

    def data_pipeline(self):

        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # create diretory to store sampling data and processed data
        base_dir = os.path.join(self.base_path, "data/train_data")
        store_dir = "data.h{}.s{}.seed{}".format(self.h_max_length, self.s_max_length, self.random_seed)
        absou_dir = os.path.join(base_dir, store_dir)
        if not os.path.exists(absou_dir):
            os.makedirs(absou_dir)

        train_data_path = os.path.join(absou_dir, "train_sample.p")
        X_train = self.train_data_loader(train_data_path, self.train_file, num_samples=self.num_negatives)
        dev_datapath = os.path.join(absou_dir, "dev_data.p")
        devs, self.dev_labels = self.dev_data_loader(dev_datapath, self.dev_file)
        test_datapath = os.path.join(absou_dir, "test_data.p")
        tests, self.test_location_indexes = self.predict_data_loader(test_datapath, self.test_file)

        self.X_train = self.train_data_tokenizer(X_train)
        self.devs = self.predict_data_tokenizer(devs)
        self.tests = self.predict_data_tokenizer(tests)

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
        doc_lines = zip(doc_lines, [page] * len(doc_lines), range(len(doc_lines)))
        return doc_lines

    def sampling(self, datapath, num_sample=1):

        jlr = JSONLineReader()

        X = []
        count = 0
        with open(datapath, "r") as f:
            lines = jlr.process(f)

            for line in tqdm(lines):
                count += 1
                pos_pairs = []
                # count1 += 1
                if line['label'].upper() == "NOT ENOUGH INFO":
                    continue
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
                    continue
                else:
                    for pos_sent in pos_set:
                        samples = random.sample(neg_sents, num_sampling)
                        for sample in samples:
                            if not sample:
                                continue
                            X.append((claim, pos_sent, sample))
                            if count % 1000 == 0:
                                print("claim:{} ,evidence :{} sample:{}".format(claim, pos_sent, sample))
        return X

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
            data = zip(devs, labels)
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
            data = zip(devs, location_indexes)
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

    def proess_sents(self, sents, max_length):

        tokenized_sents = []
        sents_lengths = []
        for sent in sents:
            words = [word.lower() for word in nltk.word_tokenize(sent)]
            if len(words) < self.h_max_length:
                sents_lengths.append(len(words))
                words.extend([""] * (self.h_max_length - len(words)))
                tokenized_sents.append(words)
            else:
                sents_lengths.append(self.h_max_length)
                words = words[:self.h_max_length]
                tokenized_sents.append(words)
        return tokenized_sents, sents_lengths

    def train_data_tokenizer(self, X_train):

        claims = [claim for claim, _, _ in X_train]
        pos_sents = [pos_sent for _, pos_sent, _ in X_train]
        neg_sents = [neg_sent for _, _, neg_sent in X_train]

        tokenized_claims, claims_lengths = self.proess_sents(claims, self.h_max_length)
        tokenized_pos_sents, pos_sents_lengths = self.proess_sents(pos_sents, self.s_max_length)
        tokenized_neg_sents, neg_sents_lengths = self.proess_sents(neg_sents, self.s_max_length)

        new_claims = list(zip(tokenized_claims, claims_lengths))
        new_pos_sents = list(zip(tokenized_pos_sents, pos_sents_lengths))
        new_neg_sents = list(zip(tokenized_neg_sents, neg_sents_lengths))

        return list(zip(new_claims, new_pos_sents, new_neg_sents))

    def predict_data_tokenizer(self, dataset):

        predict_data = []
        for data in dataset:
            claims = [claim for claim, _ in data]
            sents = [sent for _, sent in data]

            tokenized_claims, claims_lengths = self.proess_sents(claims, self.h_max_length)
            tokenized_sents, sents_lengths = self.proess_sents(sents, self.s_max_length)

            new_claims = list(zip(tokenized_claims, claims_lengths))
            new_sents = list(zip(tokenized_sents, sents_lengths))

            tokenized_data = list(zip(new_claims, new_sents))
            predict_data.append(tokenized_data)
        return predict_data
