import argparse
import json
import os
import random

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from athene.retrieval.score.score import evidence_macro_recall, fever_score
from athene.retrieval.sentences.data_processing.data import Data
from athene.retrieval.sentences.deep_models.ESIM import ESIM
from common.dataset.reader import JSONLineReader
from retrieval.fever_doc_db import FeverDocDB


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=str, help="training data path")
    parser.add_argument('--dev-data', type=str, help="development data path")
    parser.add_argument('--test-data', type=str, help="test data path", default=None)
    parser.add_argument('--out-file', type=str, help="path to save evidences")
    parser.add_argument('--num-model', type=int, help="number of models to train", default=5)
    parser.add_argument('--fasttext-path', type=str, help="fasttext model path")
    parser.add_argument('--random-seed', type=int, help="random seed for reproduction")
    parser.add_argument('--num-negatives', type=int, help="number of negative sentences for training", default=5)
    parser.add_argument('--c-max-length', type=int, help="max length for claim", default=20)
    parser.add_argument('--s-max-length', type=int, help="max length for evidence", default=60)
    parser.add_argument('--reserve-embed', type=bool, help="create more embed for unkown testing words", default=False)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--num-epoch', type=int, default=20)
    parser.add_argument('--dropout-rate', type=float, default=0.1)
    parser.add_argument('--num-lstm-units', type=float, default=128)
    parser.add_argument('--share-parameters', type=bool, default=False)
    parser.add_argument("--phase", type=str, default="training", help="either training or testing")
    parser.add_argument("--model-path", type=str, default="model/")
    args = parser.parse_args()

    return args


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


def post_processing_4_train(clf, X, indexes, k=5):
    """
    predict scores for each claim and sentences in the predicted pages,
    get the order of score in descending format, and reorder (doc_id,sent_id) pair with the order,and extract first 5 most similar sentences
    :param clf:
    :param X:
    :param indexes:C
    :param k:
    :return:
    """

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
        all_scores.append(scores[:5])
    return predictions, all_scores


def post_processing(predictions, indexes, k=5):
    """
    predict scores for each claim and sentences in the predicted pages,
    get the order of score in descending format, and reorder (doc_id,sent_id) pair with the order,and extract first 5 most similar sentences
    :param predictions:
    :param indexes:C
    :param k:
    :return:
    """

    processed_predictions = []
    all_scores = []
    for idx, prediction in tqdm(enumerate(predictions)):
        # line_input = np.asarray(line_input,np.str)
        scores = np.asarray(np.reshape(prediction, newshape=(-1,)))
        orders = np.argsort(scores, axis=-1)[::-1]
        scores = scores[orders]
        sents_indexes = np.asarray(indexes[idx])
        processed_predictions.append(sents_indexes[orders][:k])
        all_scores.append(scores[:k])
    return processed_predictions, all_scores


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
        prediction_processing_no_reload(lines, predictions)

    return final_predictions




def prediction_processing_no_reload(lines, predictions):
    """
    process the predicted (doc_id,sent_id) pairs to the score system desired format
    :param dataset_path:
    :param predictions:
    :return:
    """

    final_predictions = []
    for idx, line in enumerate(lines):
        if len(line['predicted_pages']) == 0:
            line['predicted_evidence'] = []
        else:
            line['predicted_evidence'] = [[prediction[0], int(prediction[1])] for prediction in predictions[idx]]
        line['predicted_label'] = "REFUTES"
        final_predictions.append(line)

    return final_predictions


def softmax(prediction):
    theta = 2.0
    ps = np.exp(prediction * theta)
    ps /= np.sum(ps)
    return ps


def averaging(predictions):
    processed_predictions = []
    for prediction in predictions:
        prediction = np.asarray(prediction)
        prediction = softmax(prediction)
        processed_predictions.append(prediction)
    processed_predictions = np.asarray(processed_predictions)
    final_prediction = np.mean(processed_predictions, axis=0, keepdims=False)

    return final_prediction


def scores_processing(all_predictions, args):
    ensembled_predictions = []
    for i in range(len(all_predictions[0])):
        predictions = []
        for j in range(len(all_predictions)):
            predictions.append(all_predictions[j][i])
        ensembled_prediction = averaging(predictions)
        ensembled_predictions.append(ensembled_prediction)
    return ensembled_predictions


def training_phase(path, data, args):
    random_states = [88, 12345, 4444, 8888, 9999]
    for i in range(args.num_model):
        # random_state = random.randint(0, 9999)
        # random_states.append(random_state)
        # while random_state in random_states:
        #     random_state = random.randint(0, 9999)

        model_store_path = os.path.join(args.model_path, "model{}".format(i + 1))
        os.makedirs(model_store_path, exist_ok=True)
        clf = ESIM(h_max_length=args.c_max_length, s_max_length=args.s_max_length, learning_rate=args.learning_rate,
                   batch_size=args.batch_size, num_epoch=args.num_epoch, model_store_dir=model_store_path,
                   embedding=data.embed, word_dict=data.iword_dict, dropout_rate=args.dropout_rate,
                   random_state=random_states[i], num_units=args.num_lstm_units, share_rnn=args.share_parameters,
                   activation=tf.nn.tanh)
        clf.fit(data.X_train_indexes, data.dev_indexes, data.dev_labels)

        predictions, scores = post_processing_4_train(clf, data.test_indexes, data.test_location_indexes)
        final_predictions = prediction_processing(args.dev_data, predictions)
        # show_predictions(db_filename,final_predictions)
        write_path = os.path.join(path, "data/fever/dev.esim.wiki7.exact.s5.model{}.jsonl".format(i + 1))
        write_predictions(final_predictions, write_path)

        strict_score, label_accuracy, precision, recall, f1, doc_recall = fever_score(final_predictions,
                                                                                      actual=None,
                                                                                      max_pages=None,
                                                                                      max_evidence=None)
        print("strict_score: {} label_accuracy: {} precision: {} recall: {} f1: {} doc_Recall: {}".format(
            strict_score,
            label_accuracy,
            precision, recall, f1, doc_recall))

        tf.reset_default_graph()


def prediction_phase(test_indexes, test_location_indexes, data_path, args, data, calculate_fever_score=True):
    all_predictions = []
    for i in range(args.num_model):
        model_store_path = os.path.join(args.model_path, "model{}".format(i + 1))
        if not os.path.exists(model_store_path):
            raise Exception("model must be trained before testing")
        clf = ESIM(h_max_length=args.c_max_length, s_max_length=args.s_max_length, learning_rate=args.learning_rate,
                   batch_size=args.batch_size, num_epoch=args.num_epoch, model_store_dir=model_store_path,
                   embedding=data.embed, word_dict=data.iword_dict, dropout_rate=args.dropout_rate,
                   num_units=args.num_lstm_units, share_rnn=args.share_parameters, activation=tf.nn.tanh)

        clf.restore_model(os.path.join(model_store_path, "best_model.ckpt"))
        predictions = []
        for test_index in tqdm(test_indexes):
            prediction = clf.predict(test_index)
            predictions.append(prediction)
        all_predictions.append(predictions)
        tf.reset_default_graph()

    ensembled_predicitons = scores_processing(all_predictions, args)

    processed_predictions, scores = post_processing(ensembled_predicitons, test_location_indexes)

    final_predictions = prediction_processing(data_path, processed_predictions)
    write_predictions(final_predictions, args.out_file)
    if calculate_fever_score:
        strict_score, label_accuracy, precision, recall, f1, doc_recall = fever_score(final_predictions, actual=None,
                                                                                      max_evidence=5)
        print("strict_score: {} label_accuracy: {} precision: {} recall: {} f1: {} doc_Recall: {}".format(
            strict_score,
            label_accuracy,
            precision, recall, f1, doc_recall))


def entrance(args, calculate_fever_score=True):
    random.seed(args.random_seed)
    path = os.getcwd()
    data = Data(path, args.train_data, args.dev_data, args.test_data, args.fasttext_path,
                num_negatives=args.num_negatives, h_max_length=args.c_max_length, s_max_length=args.s_max_length,
                random_seed=args.random_seed, reserve_embed=args.reserve_embed)
    if args.phase == "training":
        training_phase(path, data, args)

    elif args.phase == "deving" and args.reserve_embed:
        prediction_phase(data.test_indexes, data.test_location_indexes, args.dev_data, args, data,
                         calculate_fever_score)

    elif args.phase == "testing" and args.reserve_embed:
        data.update_word_dict(args.test_data)
        data.get_new_test_indexes(args.test_data)
        data.update_embeddings()
        prediction_phase(data.new_tests_indexes, data.new_test_location_indexes, args.test_data, args, data,
                         calculate_fever_score)

    elif args.phase == "testing" and not args.reserve_embed:
        prediction_phase(data.test_indexes, data.test_location_indexes, args.test_data, args, data,
                         calculate_fever_score)


if __name__ == "__main__":
    args = parser_args()

    entrance(args)
