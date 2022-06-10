import json
from typing import List, Union, Dict

import numpy as np
from gensim.models.wrappers import FastText
from tqdm import tqdm

from athene.rte.utils.text_processing import clean_text, vocab_map, tokenize, load_whole_glove
from common.util.log_helper import LogHelper
from common.util.random import SimpleRandom
from retrieval.fever_doc_db import FeverDocDB

label_dict = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']
dim_fasttext = 300
PAGE_LINE_SEGMENTOR = '§§§'
CLAIM = "claim"


def prediction_2_label(prediction):
    return label_dict[prediction]


def evidence_num_to_text_snopes(db: Dict, page_id: str, line: int):
    lines = db[page_id]
    if lines is None or len(lines) <= line:
        return ""
    return lines[line]


def evidence_num_to_text(db: Union[Dict, FeverDocDB], page_id: str, line: int, is_snopes: bool = False):
    assert isinstance(db, Dict) or not is_snopes, "db should be dictionary for Snopes data"
    assert isinstance(db, FeverDocDB) or is_snopes, "db should be fever DB for fever data"
    logger = LogHelper.get_logger("evidence_num_to_text")
    if is_snopes:
        return evidence_num_to_text_snopes(db, page_id, line)
    lines = db.get_doc_lines(page_id)
    if lines is None:
        return ""
    if line > -1:
        return lines.split("\n")[line].split("\t")[1]
    else:
        non_empty_lines = [line.split("\t")[1] for line in lines.split(
            "\n") if len(line.split("\t")) > 1 and len(line.split("\t")[1].strip())]
        return non_empty_lines[SimpleRandom.get_instance().next_rand(0, len(non_empty_lines) - 1)]


def ids_padding_for_multi_sentences_set(sents_list, bodies_size=None, bodies_sent_size=None):
    b_sizes = np.asarray([len(sents) for sents in sents_list])
    bodies_sent_sizes_ = [[len(sent) for sent in sents]
                          for sents in sents_list]
    if bodies_size is None:
        bodies_size = max(b_sizes)
    else:
        b_sizes = np.asarray([size if size < bodies_size else bodies_size for size in b_sizes])
    if bodies_sent_size is None:
        bodies_sent_size = max(map(max, bodies_sent_sizes_))

    def padded_text_ids(_list, sent_sizes_, num_doc, max_num_sent, max_num_words):
        doc_np = np.zeros(
            [num_doc, max_num_sent, max_num_words], dtype=np.int32)
        doc_sent_sizes = np.zeros([num_doc, max_num_sent], dtype=np.int32)
        for i, doc in enumerate(_list):
            for j, sent in enumerate(doc):
                if j >= max_num_sent:
                    break
                doc_sent_sizes[i, j] = sent_sizes_[i][j] if sent_sizes_[i][j] < max_num_words else max_num_words
                for k, word in enumerate(sent):
                    if k >= max_num_words:
                        break
                    doc_np[i, j, k] = word
        return doc_np, doc_sent_sizes

    b_np, b_sent_sizes = padded_text_ids(sents_list, bodies_sent_sizes_, len(sents_list), bodies_size,
                                         bodies_sent_size)
    return b_np, b_sizes, b_sent_sizes


def ids_padding_for_single_sentence_set_given_size(sent_list, max_sent_size=None):
    sent_sizes_ = np.asarray([len(sent) for sent in sent_list])
    if max_sent_size is None:
        max_sent_size = sent_sizes_.max()

    def padded_text_ids(_list, sent_sizes_, num_doc, max_num_words):
        doc_np = np.zeros([num_doc, max_num_words], dtype=np.int32)
        doc_sent_sizes = np.zeros([num_doc], dtype=np.int32)
        for i, doc in enumerate(_list):
            doc_sent_sizes[i] = sent_sizes_[i] if sent_sizes_[
                                                      i] < max_num_words else max_num_words
            for k, word in enumerate(doc):
                if k >= max_num_words:
                    break
                doc_np[i, k] = word
        return doc_np, doc_sent_sizes

    b_np, b_sent_sizes = padded_text_ids(
        sent_list, sent_sizes_, len(sent_list), max_sent_size)
    return b_np, b_sent_sizes


def read_data_set_from_jsonl(file_path: str, db: Union[str, FeverDocDB], predicted: bool = True, num_sentences=None,
                             is_snopes=False):
    logger = LogHelper.get_logger("read_data_set_from_jsonl")
    if not is_snopes:
        if type(db) is str:
            db = FeverDocDB(db)
    else:
        with open(db) as f:
            db = json.load(f)
    with open(file_path, 'r') as f:
        claims = []
        evidences = []
        paths = []
        labels = []
        ids = []
        for line in tqdm(f):
            json_obj = json.loads(line)
            if predicted:
                evidences_texts = []
                if 'predicted_evidence' in json_obj:
                    _evidences = json_obj['predicted_evidence']
                elif 'predicted_sentences' in json_obj:
                    _evidences = json_obj['predicted_sentences']
                else:
                    _evidences = []
                if len(_evidences) > 0:
                    for sent in _evidences:
                        page, line_num = sent[-2], sent[-1]
                        page_title = page.replace("_", " ")
                        evidences_texts.append(
                            # page_title + " # " + clean_text(evidence_num_to_text(db, page, line_num, is_snopes)))
                            clean_text(evidence_num_to_text(db, page, line_num, is_snopes)))
            else:
                evidences_texts = set()
                _evidences = json_obj['evidence']
                for evidence in _evidences:
                    for sent in evidence:
                        page, line_num = sent[-2], sent[-1]
                        page_title = page.replace("_", " ")
                        evidences_texts.add(
                            # page_title + " # " + clean_text(evidence_num_to_text(db, page, line_num, is_snopes)))
                            clean_text(evidence_num_to_text(db, page, line_num, is_snopes)))
                evidences_texts = list(evidences_texts)
            if len(evidences_texts) == 0:
                continue
            if num_sentences is not None:
                if len(evidences_texts) > num_sentences:
                    evidences_texts = evidences_texts[:num_sentences]
            claims.append(clean_text(json_obj['claim']))
            if 'label' in json_obj:
                labels.append(label_dict.index(json_obj['label']))
            evidences.append(evidences_texts)
            if 'paths' in json_obj:
                paths_from_sent_to_claim = [1.0 if p else 0.0 for p in json_obj['paths']]
                if num_sentences is not None and num_sentences > len(paths_from_sent_to_claim):
                    paths_from_sent_to_claim += [0.0] * (num_sentences - len(paths_from_sent_to_claim))
                paths.append(paths_from_sent_to_claim)
            ids.append(json_obj['id'])
        datas = {'h': claims, 'b': evidences, 'id': ids}
        if paths:
            datas['paths'] = paths
        return datas, labels



def read_data_set_from_lines(lines: List, db: Union[str, FeverDocDB], predicted: bool = True, num_sentences=None,
                             is_snopes=False):
    logger = LogHelper.get_logger("read_data_set_from_jsonl")
    if not is_snopes:
        if type(db) is str:
            db = FeverDocDB(db)
    else:
        with open(db) as f:
            db = json.load(f)

    claims = []
    evidences = []
    paths = []
    labels = []

    for line in tqdm(lines):
        json_obj = line
        if predicted:
            evidences_texts = []
            if 'predicted_evidence' in json_obj:
                _evidences = json_obj['predicted_evidence']
            elif 'predicted_sentences' in json_obj:
                _evidences = json_obj['predicted_sentences']
            else:
                _evidences = []
            if len(_evidences) > 0:
                for sent in _evidences:
                    page, line_num = sent[-2], sent[-1]
                    page_title = page.replace("_", " ")
                    evidences_texts.append(
                        # page_title + " # " + clean_text(evidence_num_to_text(db, page, line_num, is_snopes)))
                        clean_text(evidence_num_to_text(db, page, line_num, is_snopes)))
        else:
            evidences_texts = set()
            _evidences = json_obj['evidence']
            for evidence in _evidences:
                for sent in evidence:
                    page, line_num = sent[-2], sent[-1]
                    page_title = page.replace("_", " ")
                    evidences_texts.add(
                        # page_title + " # " + clean_text(evidence_num_to_text(db, page, line_num, is_snopes)))
                        clean_text(evidence_num_to_text(db, page, line_num, is_snopes)))
            evidences_texts = list(evidences_texts)

        if len(evidences_texts) == 0:
            evidence_texts = [""]

        if num_sentences is not None:
            if len(evidences_texts) > num_sentences:
                evidences_texts = evidences_texts[:num_sentences]
        claims.append(clean_text(json_obj['claim']))
        if 'label' in json_obj:
            labels.append(label_dict.index(json_obj['label']))
        evidences.append(evidences_texts)
        if 'paths' in json_obj:
            paths_from_sent_to_claim = [1.0 if p else 0.0 for p in json_obj['paths']]
            if num_sentences is not None and num_sentences > len(paths_from_sent_to_claim):
                paths_from_sent_to_claim += [0.0] * (num_sentences - len(paths_from_sent_to_claim))
            paths.append(paths_from_sent_to_claim)

    datas = {'h': claims, 'b': evidences}
    if paths:
        datas['paths'] = paths
    return datas, labels

def generate_evidence_labels(_evidences, _gold_evidences):
    _segmenter = "§§§"
    _gold_sents = set()
    for _gold_evidence in _gold_evidences:
        for _gold_sent in _gold_evidence:
            page, sent = _gold_sent[-2], _gold_sent[-1]
            if page is not None and sent is not None:
                _gold_sents.add(page + _segmenter + str(sent))
    _labels = []
    for _pred_sent in _evidences:
        key = _pred_sent[-2] + _segmenter + str(_pred_sent[-1])
        if key in _gold_sents:
            _labels.append(1)
        else:
            _labels.append(0)
    return _labels


def single_sentence_set_2_ids_given_vocab(texts, vocab_dict):
    logger = LogHelper.get_logger("single_sentence_set_2_ids_given_vocab")
    doc_ids = []
    out_of_vocab_counts = 0
    for sent in texts:
        tokens = tokenize(sent)
        word_ids = []
        for token in tokens:
            if token.lower() in vocab_dict:
                word_ids.append(vocab_dict[token.lower()])
            else:
                out_of_vocab_counts += 1
                word_ids.append(vocab_dict['UNK'])
        doc_ids.append(word_ids)
    logger.debug("{} times out of vocab".format(str(out_of_vocab_counts)))
    return doc_ids


def multi_sentence_set_2_ids_given_vocab(texts, vocab_dict):
    logger = LogHelper.get_logger("multi_sentence_set_2_ids_given_vocab")
    doc_ids = []
    out_of_vocab_counts = 0
    for sents in texts:
        sent_ids = []
        for sent in sents:
            tokens = tokenize(sent)
            word_ids = []
            for token in tokens:
                if token.lower() in vocab_dict:
                    word_ids.append(vocab_dict[token.lower()])
                else:
                    word_ids.append(vocab_dict['UNK'])
            sent_ids.append(word_ids)
        doc_ids.append(sent_ids)
    logger.debug("{} times out of vocab".format(str(out_of_vocab_counts)))
    return doc_ids


def single_sentence_set_2_fasttext_embedded(sents: List[str], fasttext_model: Union[str, FastText]):
    logger = LogHelper.get_logger("single_sentence_set_2_fasttext_embedded")
    if type(fasttext_model) == str:
        fasttext_model = FastText.load_fasttext_format(fasttext_model)
    fasttext_embeddings = []
    for sent in sents:
        tokens = tokenize(sent)
        sent_embeddings = []
        for token in tokens:
            try:
                sent_embeddings.append(fasttext_model[token.lower()])
            except KeyError:
                sent_embeddings.append(np.ones([dim_fasttext], np.float32))
        fasttext_embeddings.append(sent_embeddings)
    return fasttext_embeddings, fasttext_model


def multi_sentence_set_2_fasttext_embedded(texts: List[List[str]], fasttext_model: Union[str, FastText]):
    logger = LogHelper.get_logger("multi_sentence_set_2_fasttext_embedded")
    if type(fasttext_model) == str:
        fasttext_model = FastText.load_fasttext_format(fasttext_model)
    fasttext_embeddings = []
    for sents in texts:
        text_embeddings = []
        for sent in sents:
            tokens = tokenize(sent)
            sent_embeddings = []
            for token in tokens:
                try:
                    sent_embeddings.append(fasttext_model[token.lower()])
                except KeyError:
                    sent_embeddings.append(np.ones([dim_fasttext], np.float32))
            text_embeddings.append(sent_embeddings)
        fasttext_embeddings.append(text_embeddings)
    return fasttext_embeddings, fasttext_model


def fasttext_padding_for_single_sentence_set_given_size(fasttext_embeddings, max_sent_size=None):
    logger = LogHelper.get_logger("fasttext_padding_for_single_sentence_set_given_size")
    sent_sizes_ = np.asarray([len(sent) for sent in fasttext_embeddings])
    if max_sent_size is None:
        max_sent_size = sent_sizes_.max()

    def padded_text_ids(_list, num_doc, max_num_words):
        doc_np = np.zeros([num_doc, max_num_words, dim_fasttext], dtype=np.float32)
        for i, doc in enumerate(_list):
            for k, word in enumerate(doc):
                if k >= max_num_words:
                    break
                doc_np[i, k] = word
        return doc_np

    ft_np = padded_text_ids(fasttext_embeddings, len(fasttext_embeddings), max_sent_size)
    return ft_np


def fasttext_padding_for_multi_sentences_set(fasttext_embeddings, max_bodies_size=None, max_bodies_sent_size=None):
    logger = LogHelper.get_logger("fasttext_padding_for_multi_sentences_set")
    b_sizes = np.asarray([len(sents) for sents in fasttext_embeddings])
    bodies_sent_sizes_ = [[len(sent) for sent in sents] for sents in fasttext_embeddings]
    if max_bodies_size is None:
        max_bodies_size = max(b_sizes)
    if max_bodies_sent_size is None:
        max_bodies_sent_size = max(map(max, bodies_sent_sizes_))

    def padded_text_ids(_list, num_doc, max_num_sent, max_num_words):
        doc_np = np.zeros([num_doc, max_num_sent, max_num_words, dim_fasttext], dtype=np.float32)
        for i, doc in enumerate(_list):
            for j, sent in enumerate(doc):
                if j >= max_num_sent:
                    break
                for k, word in enumerate(sent):
                    if k >= max_num_words:
                        break
                    doc_np[i, j, k] = word
        return doc_np

    ft_np = padded_text_ids(fasttext_embeddings, len(fasttext_embeddings), max_bodies_size, max_bodies_sent_size)
    return ft_np


def embed_data_set_with_glove_and_fasttext(data_set_path: str, db: Union[str, FeverDocDB],
                                           fasttext_model: Union[str, FastText], glove_path: str = None,
                                           vocab_dict: Dict[str, int] = None, glove_embeddings=None,
                                           predicted: bool = True, threshold_b_sent_num=None,
                                           threshold_b_sent_size=50, threshold_h_sent_size=50, is_snopes=False):
    assert vocab_dict is not None and glove_embeddings is not None or glove_path is not None, "Either vocab_dict and glove_embeddings, or glove_path should be not None"
    if vocab_dict is None or glove_embeddings is None:
        vocab, glove_embeddings = load_whole_glove(glove_path)
        vocab_dict = vocab_map(vocab)
    logger = LogHelper.get_logger("embed_data_set_given_vocab")
    datas, labels = read_data_set_from_jsonl(data_set_path, db, predicted, is_snopes=is_snopes)
    heads_ft_embeddings, fasttext_model = single_sentence_set_2_fasttext_embedded(datas['h'], fasttext_model)
    logger.debug("Finished sentence to FastText embeddings for claims")
    heads_ids = single_sentence_set_2_ids_given_vocab(datas['h'], vocab_dict)
    logger.debug("Finished sentence to IDs for claims")
    bodies_ft_embeddings, fasttext_model = multi_sentence_set_2_fasttext_embedded(datas['b'], fasttext_model)
    logger.debug("Finished sentence to FastText embeddings for evidences")
    bodies_ids = multi_sentence_set_2_ids_given_vocab(datas['b'], vocab_dict)
    logger.debug("Finished sentence to IDs for evidences")
    h_ft_np = fasttext_padding_for_single_sentence_set_given_size(heads_ft_embeddings, threshold_h_sent_size)
    logger.debug("Finished padding FastText embeddings for claims. Shape of h_ft_np: {}".format(str(h_ft_np.shape)))
    b_ft_np = fasttext_padding_for_multi_sentences_set(bodies_ft_embeddings, threshold_b_sent_num,
                                                       threshold_b_sent_size)
    logger.debug("Finished padding FastText embeddings for evidences. Shape of b_ft_np: {}".format(str(b_ft_np.shape)))
    h_np, h_sent_sizes = ids_padding_for_single_sentence_set_given_size(
        heads_ids, threshold_h_sent_size)
    logger.debug("Finished padding claims")
    b_np, b_sizes, b_sent_sizes = ids_padding_for_multi_sentences_set(
        bodies_ids, threshold_b_sent_num, threshold_b_sent_size)
    logger.debug("Finished padding evidences")
    processed_data_set = {'data': {
        'h_np': h_np,
        'b_np': b_np,
        'h_ft_np': h_ft_np,
        'b_ft_np': b_ft_np,
        'h_sent_sizes': h_sent_sizes,
        'b_sent_sizes': b_sent_sizes,
        'b_sizes': b_sizes
    }, 'id': datas['id']
    }
    if labels is not None and len(labels) == len(processed_data_set['id']):
        processed_data_set['label'] = labels
    return processed_data_set, fasttext_model, vocab_dict, glove_embeddings, threshold_b_sent_num, threshold_b_sent_size


def embed_claims(claims: List, db: Union[str, FeverDocDB],
                                           fasttext_model: Union[str, FastText], glove_path: str = None,
                                           vocab_dict: Dict[str, int] = None, glove_embeddings=None,
                                           predicted: bool = True, threshold_b_sent_num=None,
                                           threshold_b_sent_size=50, threshold_h_sent_size=50, is_snopes=False):
    assert vocab_dict is not None and glove_embeddings is not None or glove_path is not None, "Either vocab_dict and glove_embeddings, or glove_path should be not None"
    if vocab_dict is None or glove_embeddings is None:
        vocab, glove_embeddings = load_whole_glove(glove_path)
        vocab_dict = vocab_map(vocab)
    print(len(claims))
    logger = LogHelper.get_logger("embed_data_set_given_vocab")
    datas, labels = read_data_set_from_lines(claims, db, predicted, is_snopes=is_snopes)
    print(len(datas["h"]),len(datas["b"]))
    heads_ft_embeddings, fasttext_model = single_sentence_set_2_fasttext_embedded(datas['h'], fasttext_model)
    logger.debug("Finished sentence to FastText embeddings for claims")
    print(len(heads_ft_embeddings))
    heads_ids = single_sentence_set_2_ids_given_vocab(datas['h'], vocab_dict)
    logger.debug("Finished sentence to IDs for claims")
    bodies_ft_embeddings, fasttext_model = multi_sentence_set_2_fasttext_embedded(datas['b'], fasttext_model)
    logger.debug("Finished sentence to FastText embeddings for evidences")
    bodies_ids = multi_sentence_set_2_ids_given_vocab(datas['b'], vocab_dict)
    logger.debug("Finished sentence to IDs for evidences")
    h_ft_np = fasttext_padding_for_single_sentence_set_given_size(heads_ft_embeddings, threshold_h_sent_size)
    logger.debug("Finished padding FastText embeddings for claims. Shape of h_ft_np: {}".format(str(h_ft_np.shape)))
    b_ft_np = fasttext_padding_for_multi_sentences_set(bodies_ft_embeddings, threshold_b_sent_num,
                                                       threshold_b_sent_size)
    logger.debug("Finished padding FastText embeddings for evidences. Shape of b_ft_np: {}".format(str(b_ft_np.shape)))
    h_np, h_sent_sizes = ids_padding_for_single_sentence_set_given_size(
        heads_ids, threshold_h_sent_size)
    logger.debug("Finished padding claims")
    b_np, b_sizes, b_sent_sizes = ids_padding_for_multi_sentences_set(
        bodies_ids, threshold_b_sent_num, threshold_b_sent_size)
    logger.debug("Finished padding evidences")
    processed_data_set = {'data': {
        'h_np': h_np,
        'b_np': b_np,
        'h_ft_np': h_ft_np,
        'b_ft_np': b_ft_np,
        'h_sent_sizes': h_sent_sizes,
        'b_sent_sizes': b_sent_sizes,
        'b_sizes': b_sizes
    }
    }

    return processed_data_set, fasttext_model, vocab_dict, glove_embeddings, threshold_b_sent_num, threshold_b_sent_size



def pad_paths(paths, threshold_b_sent_num):
    padded_paths = []
    for path_of_claim in paths:
        _paths = path_of_claim[:threshold_b_sent_num]
        if len(_paths) < threshold_b_sent_num:
            _paths += [0.0] * (threshold_b_sent_num - len(_paths))
        padded_paths.append(_paths)
    return np.asarray(padded_paths, np.float32)


def embed_data_set_with_glove_2(data_set_path: str, db: Union[str, FeverDocDB], glove_path: str = None,
                                vocab_dict: Dict[str, int] = None, glove_embeddings=None, predicted: bool = True,
                                threshold_b_sent_num=None, threshold_b_sent_size=50, threshold_h_sent_size=50):
    if vocab_dict is None or glove_embeddings is None:
        vocab, glove_embeddings = load_whole_glove(glove_path)
        vocab_dict = vocab_map(vocab)
    logger = LogHelper.get_logger("embed_data_set_given_vocab")
    datas, labels = read_data_set_from_jsonl(data_set_path, db, predicted)
    heads_ids = single_sentence_set_2_ids_given_vocab(datas['h'], vocab_dict)
    logger.debug("Finished sentence to IDs for claims")
    bodies_ids = multi_sentence_set_2_ids_given_vocab(datas['b'], vocab_dict)
    logger.debug("Finished sentence to IDs for evidences")
    h_np, h_sent_sizes = ids_padding_for_single_sentence_set_given_size(
        heads_ids, threshold_h_sent_size)
    logger.debug("Finished padding claims")
    b_np, b_sizes, b_sent_sizes = ids_padding_for_multi_sentences_set(
        bodies_ids, threshold_b_sent_num, threshold_b_sent_size)
    logger.debug("Finished padding evidences")
    processed_data_set = {'data': {
        'h_np': h_np,
        'b_np': b_np,
        'h_sent_sizes': h_sent_sizes,
        'b_sent_sizes': b_sent_sizes,
        'b_sizes': b_sizes
    }, 'id': datas['id']
    }
    if 'paths' in datas:
        padded_paths_np = pad_paths(datas['paths'], threshold_b_sent_num)
        processed_data_set['data']['paths'] = padded_paths_np
    if labels is not None and len(labels) == len(processed_data_set['id']):
        processed_data_set['label'] = labels
    return processed_data_set, vocab_dict, glove_embeddings, threshold_b_sent_num, threshold_b_sent_size


def _concat_sent(_page, _line_num):
    return _page + PAGE_LINE_SEGMENTOR + str(_line_num)


def _split_sent_str(sent_str):
    segments = sent_str.split(PAGE_LINE_SEGMENTOR)
    assert len(segments) == 2, "invalid page line_num concatenation: " + sent_str
    return segments[0], int(segments[1])


def load_feature_by_data_set(data_set_path: str, feature_path: str, max_sent_num: int):
    from common.dataset.reader import JSONLineReader
    import pickle
    import os
    with open(os.path.join(feature_path, 'feature.p'), 'rb') as f:
        features = pickle.load(f)
    with open(os.path.join(feature_path, 'data_idx_map.p'), 'rb') as f:
        data_idx_map = pickle.load(f)
    jlr = JSONLineReader()
    lines = jlr.read(data_set_path)
    feature_dim = features.shape[1]
    padding = np.zeros([feature_dim], np.float32)
    claim_features = []
    evidence_features = []
    for line in lines:
        _id = line['id']
        key = _concat_sent(CLAIM, _id)
        claim_features.append(features[data_idx_map[key]])
        evidence_per_claim_features = []
        for sent in line['predicted_evidence']:
            page, line_num = sent[-2], sent[-1]
            key = _concat_sent(page, line_num)
            evidence_per_claim_features.append(features[data_idx_map[key]])
        if len(evidence_per_claim_features) > max_sent_num:
            evidence_features.append(evidence_per_claim_features[:max_sent_num])
        else:
            for _ in range(max_sent_num - len(evidence_per_claim_features)):
                evidence_per_claim_features.append(padding)
            evidence_features.append(evidence_per_claim_features)
    return np.asarray(claim_features, np.float32), np.asarray(evidence_features, np.float32)


def is_token_numeric(token):
    import re
    return re.fullmatch(r'[+-]?((\d+(\.\d*)?)|(\.\d+))', token)


def _interprete_num_result(has_num, has_identical_num, has_different_num):
    if not has_num:
        return [0, 0, 0]
    result = [1, 0, 0]
    if has_identical_num:
        result[1] = 1
    if has_different_num:
        result[2] = 1
    return result


def number_feature(data_set_path: str, db_path: str, max_sent_num: int):
    from common.dataset.reader import JSONLineReader
    db = FeverDocDB(db_path)
    jlr = JSONLineReader()
    lines = jlr.read(data_set_path)
    num_feat = np.zeros([len(lines), max_sent_num, 3], dtype=np.int32)
    for i, line in enumerate(lines):
        claim_text = line['claim']
        claim_tokens = tokenize(claim_text)
        all_nums = set()
        for token in claim_tokens:
            if is_token_numeric(token):
                all_nums.add(float(token))
        for j, evidence in enumerate(line['predicted_evidence']):
            if j >= max_sent_num:
                break
            page, line_num = evidence[-2], evidence[-1]
            all_evidence_nums = []
            evidence_text = evidence_num_to_text(db, page, line_num)
            evidence_tokens = tokenize(evidence_text)
            for token in evidence_tokens:
                if is_token_numeric(token):
                    all_evidence_nums.append(float(token))
            has_num = len(all_evidence_nums) > 0
            has_identical_num = any(n in all_nums for n in all_evidence_nums)
            has_different_num = any(n not in all_nums for n in all_evidence_nums)
            num_feat[i][j][0], num_feat[i][j][1], num_feat[i][j][2] = _interprete_num_result(has_num,
                                                                                             has_identical_num,
                                                                                             has_different_num)
    return num_feat


def generate_concat_indices_for_inter_evidence(evidences_np, evidences_sizes_np, max_sent_size: int, max_sent_num: int):
    batch_size = evidences_np.shape[0]
    concat_indices = []
    concat_sent_sizes = []
    for i in range(batch_size):
        concat_indices_per_claim = []
        concat_sent_sizes_per_claim = []
        for j in range(max_sent_num):
            token_indices_per_sent = np.array([], dtype=np.int32)
            padding_indices_per_sent = np.array([], dtype=np.int32)
            for k in range(max_sent_num):
                if j == k:
                    continue
                start = max_sent_size * k
                token_indices = np.arange(start, start + evidences_sizes_np[i][k], dtype=np.int32)
                padding_indices = np.arange(start + evidences_sizes_np[i][k], start + max_sent_size, dtype=np.int32)
                token_indices_per_sent = np.concatenate((token_indices_per_sent, token_indices))
                padding_indices_per_sent = np.concatenate((padding_indices_per_sent, padding_indices))
            concat_indices_per_sent = np.concatenate((token_indices_per_sent, padding_indices_per_sent))
            concat_indices_per_claim.append(concat_indices_per_sent)
            concat_sent_size = 0 if evidences_sizes_np[i][j] == 0 else token_indices_per_sent.shape[0]
            concat_sent_sizes_per_claim.append(concat_sent_size)
        concat_indices.append(concat_indices_per_claim)
        concat_sent_sizes.append(concat_sent_sizes_per_claim)
    return np.asarray(concat_indices), np.asarray(concat_sent_sizes)


def generate_concat_indices_for_claim(evidences_np, evidences_sizes_np, max_sent_size: int, max_sent_num: int):
    batch_size = evidences_np.shape[0]
    concat_indices = []
    concat_sent_sizes = []
    for i in range(batch_size):
        concat_indices_per_claim = []
        token_indices_per_claim = np.array([], dtype=np.int32)
        padding_indices_per_claim = np.array([], dtype=np.int32)
        for k in range(max_sent_num):
            start = max_sent_size * k
            token_indices = np.arange(start, start + evidences_sizes_np[i][k], dtype=np.int32)
            padding_indices = np.arange(start + evidences_sizes_np[i][k], start + max_sent_size, dtype=np.int32)
            token_indices_per_claim = np.concatenate((token_indices_per_claim, token_indices))
            padding_indices_per_claim = np.concatenate((padding_indices_per_claim, padding_indices))
            concat_indices_per_claim = np.concatenate((token_indices_per_claim, padding_indices_per_claim))
        concat_sent_sizes_per_claim = token_indices_per_claim.shape[0]
        concat_indices.append(concat_indices_per_claim)
        concat_sent_sizes.append(concat_sent_sizes_per_claim)
    return np.asarray(concat_indices), np.asarray(concat_sent_sizes)
