import json
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from athene.retrieval.score.score import fever_score, evidence_macro_recall
from athene.retrieval.sentences.data_processing.elmo_data import ELMO_Data
from athene.retrieval.sentences.deep_models.ESIM import ESIM
from athene.retrieval.sentences.deep_models.ESIMandELMO import ELMO_ESIM
from common.dataset.reader import JSONLineReader
from retrieval.fever_doc_db import FeverDocDB


def test_data(db_path, dataset_path, type="ranking"):
    """
    generate dev examples to feed into the classifier
    :param db_path:
    :param dataset_path:
    :param type:
    :return:
    """

    db = FeverDocDB(db_path)
    jsr = JSONLineReader()

    inputs = []
    X_claim = []
    X_sents = []
    indexes = []

    with open(dataset_path, "r") as f:
        lines = jsr.process(f)

        for line in tqdm(lines):

            p_lines = []
            valid_lines = []
            claims = []
            sents_idnexes = []
            claim = line['claim']
            # X_claim.append([claim])
            predicted_pages = line['predicted_pages']
            for page in predicted_pages:
                # doc_lines = db.get_doc_lines(page[0])
                doc_lines = db.get_doc_lines(page[0])

                if not doc_lines:
                    # print(page)
                    continue
                doc_lines = [doc_line.split("\t")[1] if len(doc_line.split("\t")[1]) > 1 else "" for doc_line in
                             doc_lines.split("\n")]
                p_lines.extend(zip(doc_lines, [page[0]] * len(doc_lines), range(len(doc_lines))))

            for doc_line in p_lines:
                if not doc_line[0]:
                    continue
                else:
                    # print(doc_line[0])
                    if type == "cos":
                        sents_idnexes.append((doc_line[1], doc_line[2]))
                        valid_lines.append(doc_line[0])
                        claims.append(claim)
                    elif type == "ranking":
                        sents_idnexes.append((doc_line[1], doc_line[2]))
                        valid_lines.append((claim, doc_line[0]))
            if type == "cos":
                X_sents.append(valid_lines)
                X_claim.append(claims)
            elif type == "ranking":
                inputs.append(valid_lines)
            indexes.append(sents_idnexes)
        inputs = list(zip(X_claim, X_sents))

        return inputs, indexes


def write_predictions(final_predictions, write_path):
    with open(write_path, "w+") as f:
        for prediction in final_predictions:
            f.write(json.dumps(prediction) + "\n")


def show_predictions(db_filename, predictions):
    """
    display claim and predicted sentences which doesn't include at least one evidence set
    :param db_filename:
    :param predictions:
    :return:
    """

    db = FeverDocDB(db_filename)

    for line in predictions:

        if line['label'].upper() != "NOT ENOUGH INFO":
            macro_rec = evidence_macro_recall(line)
            if macro_rec[0] == 1.0:
                continue
            pages = set([page for page, _ in line['predicted_evidence']])
            evidence_set = set([(page, line_num) for page, line_num in line['predicted_evidence']])
            p_lines = []
            for page in pages:
                doc_lines = db.get_doc_lines(page)
                if not doc_lines:
                    # print(page)
                    continue
                doc_lines = [doc_line.split("\t")[1] if len(doc_line.split("\t")[1]) > 1 else "" for doc_line in
                             doc_lines.split("\n")]
                p_lines.extend(zip(doc_lines, [page] * len(doc_lines), range(len(doc_lines))))

            print("claim: {}".format(line['claim']))
            print(evidence_set)
            count = 0
            for doc_line in p_lines:
                if (doc_line[1], doc_line[2]) in evidence_set:
                    print("the {}st evidence: {}".format(count, doc_line[0]))
                    count += 1


def post_processing(clf, X, indexes, k=50, type="siamese"):
    predictions = []
    all_scores = []
    for idx, line_input in tqdm(enumerate(X)):
        # line_input = np.asarray(line_input,np.str)
        scores = clf.predict(line_input)
        scores = np.asarray(np.reshape(scores, newshape=(-1,)))
        orders = np.argsort(scores, axis=-1)[::-1]
        scores = scores[orders]
        sents_indexes = np.asarray(indexes[idx])
        predictions.append(sents_indexes[orders][:k])
        all_scores.append(scores[:k])
    return predictions, all_scores


def prediction_processing(dataset_path, predictions):
    """
    process the predicted (doc_id,sent_id) pairs to the score system desired format
    :param dataset_path:
    :param predictions:
    :return:
    """

    final_predictions = []
    jsr = JSONLineReader()

    with open(dataset_path, "r") as f:
        lines = jsr.process(f)
        #
        # lines = lines[:100]

        for idx, line in enumerate(lines):
            line['predicted_evidence'] = [[prediction[0], int(prediction[1])] for prediction in predictions[idx]]
            line['predicted_label'] = "refutes"
            final_predictions.append(line)

    return final_predictions


def main(model="use_cosine", model_store_dir="model/sentence_retrieval", training_set="train.wiki7.jsonl",
         dev_set="dev.wiki7.jsonl", test_set="test.wiki7.jsonl", output_test_set="data/fever/test.p7.s5.jsonl",
         embedding_path=None, mode="train", clf=None):
    """
    the pipeline to store different kind of models, and their different kind of inputs
    :param model:
    :return:
    """

    path = os.getcwd()
    dev_path = os.path.join(path, dev_set)
    test_path = os.path.join(path, test_set)
    new_train_path = os.path.join(path, training_set)
    fasttext_path = os.path.join(path, "data/fasttext/wiki.en.bin")
    restore_param_required = clf is None
    if embedding_path is None:
        embedding_path = path

    if model == "esim":
        from athene.retrieval.sentences.data_processing.data import Data
        data = Data(embedding_path, new_train_path, dev_path, test_path, fasttext_path, num_negatives=5,
                    h_max_length=20, s_max_length=20, random_seed=100,
                    db_filepath=os.path.join(path, "data/fever/fever.db"))
        if mode == "data_preprocessing":
            return
        if clf is None:
            clf = ESIM(h_max_length=20, s_max_length=20, learning_rate=0.001, batch_size=256, num_epoch=20,
                       model_store_dir=model_store_dir, embedding=data.embed, word_dict=data.iword_dict,
                       dropout_rate=0.2, random_state=88, num_units=128, activation=tf.nn.relu, share_rnn=True)
        if mode == "train":
            clf.fit(data.X_train_indexes, data.dev_indexes, data.dev_labels)
        elif restore_param_required:
            clf.restore_model(os.path.join(model_store_dir, "best_model.ckpt"))
        # print(data.test_location_indexes[0])
        predictions, scores = post_processing(clf, data.test_indexes, data.test_location_indexes)

        final_predictions = prediction_processing(test_path, predictions)
        # show_predictions(db_filename,final_predictions)
        write_path = os.path.join(path, output_test_set)
        write_predictions(final_predictions, write_path)
        if mode == "train":
            strict_score, label_accuracy, precision, recall, f1, doc_recall = fever_score(final_predictions,
                                                                                          actual=None,
                                                                                          max_pages=None,
                                                                                          max_evidence=None)
            print("strict_score: {} label_accuracy: {} precision: {} recall: {} f1: {} doc_Recall: {}".format(
                strict_score,
                label_accuracy,
                precision, recall, f1, doc_recall))

        # tf.reset_default_graph()

    elif model == "elmo+esim":
        data = ELMO_Data(embedding_path, new_train_path, dev_path, test_path, num_negatives=5, h_max_length=20,
                         s_max_length=60, random_seed=100, db_filepath=os.path.join(path, "data/fever/fever.db"))
        if mode == "data_preprocessing":
            return
        if clf is None:
            clf = ELMO_ESIM(learning_rate=0.001, batch_size=128, num_epoch=10, model_store_dir=model_store_dir,
                            dropout_rate=0.1, random_state=88, num_units=128, activation=tf.nn.tanh)
        if mode == "train":
            clf.fit(data.X_train, data.devs, data.dev_labels)
        elif restore_param_required:
            clf.restore_model(os.path.join(model_store_dir, "esim_elmo_best_model.ckpt"))
        predictions, scores = post_processing(clf, data.tests, data.test_location_indexes)

        final_predictions = prediction_processing(test_path, predictions)
        # show_predictions(db_filename,final_predictions)
        write_path = os.path.join(path, output_test_set)
        write_predictions(final_predictions, write_path)
        if mode == "train":
            strict_score, label_accuracy, precision, recall, f1, doc_recall = fever_score(final_predictions,
                                                                                          actual=None,
                                                                                          max_evidence=5)
            print("strict_score: {} label_accuracy: {} precision: {} recall: {} f1: {} doc_Recall: {}".format(
                strict_score,
                label_accuracy,
                precision,
                recall,
                f1,
                doc_recall))
    return clf


if __name__ == '__main__':
    main(model="esim")
