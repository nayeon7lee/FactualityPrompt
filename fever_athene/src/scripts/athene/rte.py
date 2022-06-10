import argparse
import os
import pickle

import numpy as np

from athene.rte.utils.data_reader import embed_data_set_with_glove_2, load_feature_by_data_set, \
    number_feature, generate_concat_indices_for_inter_evidence, generate_concat_indices_for_claim
from athene.rte.utils.estimator_definitions import get_estimator
from athene.rte.utils.score import print_metrics
from athene.rte.utils.text_processing import load_whole_glove, vocab_map
from athene.utils.config import Config
from common.util.log_helper import LogHelper
from scripts.athene.rte_fasttext import main as main_fasttext
from .rte_fasttext import save_model, load_model, generate_submission


def main(mode, config, estimator=None):
    LogHelper.setup()
    logger = LogHelper.get_logger(os.path.splitext(os.path.basename(__file__))[0] + "_" + mode)
    logger.info("model: " + mode + ", config: " + str(config))
    if hasattr(Config, 'use_inter_evidence_comparison'):
        use_inter_evidence_comparison = Config.use_inter_evidence_comparison
    else:
        use_inter_evidence_comparison = False
    if hasattr(Config, 'use_claim_evidences_comparison'):
        use_claim_evidences_comparison = Config.use_claim_evidences_comparison
    else:
        use_claim_evidences_comparison = False
    if hasattr(Config, 'use_extra_features'):
        use_extra_features = Config.use_extra_features
    else:
        use_extra_features = False
    if hasattr(Config, 'use_numeric_feature'):
        use_numeric_feature = Config.use_numeric_feature
    else:
        use_numeric_feature = False
    logger.info("scorer type: " + Config.estimator_name)
    logger.info("random seed: " + str(Config.seed))
    logger.info("ESIM arguments: " + str(Config.esim_end_2_end_hyper_param))
    logger.info("use_inter_sentence_comparison: " + str(use_inter_evidence_comparison))
    logger.info("use_extra_features: " + str(use_extra_features))
    logger.info("use_numeric_feature: " + str(use_numeric_feature))
    logger.info("use_claim_evidences_comparison: " + str(use_claim_evidences_comparison))
    if mode == 'train':
        # # training mode
        if hasattr(Config, 'training_dump') and os.path.exists(Config.training_dump):
            with open(Config.training_dump, 'rb') as f:
                (X_dict, y_train) = pickle.load(f)
        else:
            training_set, vocab, embeddings, _, _ = embed_data_set_with_glove_2(Config.training_set_file,
                                                                                Config.db_path,
                                                                                glove_path=Config.glove_path,
                                                                                threshold_b_sent_num=Config.max_sentences,
                                                                                threshold_b_sent_size=Config.max_sentence_size,
                                                                                threshold_h_sent_size=Config.max_claim_size)
            h_sent_sizes = training_set['data']['h_sent_sizes']
            h_sizes = np.ones(len(h_sent_sizes), np.int32)
            training_set['data']['h_sent_sizes'] = np.expand_dims(h_sent_sizes, 1)
            training_set['data']['h_sizes'] = h_sizes
            training_set['data']['h_np'] = np.expand_dims(training_set['data']['h_np'], 1)

            valid_set, _, _, _, _ = embed_data_set_with_glove_2(Config.dev_set_file, Config.db_path,
                                                                vocab_dict=vocab, glove_embeddings=embeddings,
                                                                threshold_b_sent_num=Config.max_sentences,
                                                                threshold_b_sent_size=Config.max_sentence_size,
                                                                threshold_h_sent_size=Config.max_claim_size)
            h_sent_sizes = valid_set['data']['h_sent_sizes']
            h_sizes = np.ones(len(h_sent_sizes), np.int32)
            valid_set['data']['h_sent_sizes'] = np.expand_dims(h_sent_sizes, 1)
            valid_set['data']['h_sizes'] = h_sizes
            valid_set['data']['h_np'] = np.expand_dims(valid_set['data']['h_np'], 1)
            if use_extra_features:
                assert hasattr(Config, 'feature_path'), "Config should has feature_path if Config.use_feature is True"
                training_claim_features, training_evidence_features = load_feature_by_data_set(Config.training_set_file,
                                                                                               Config.feature_path,
                                                                                               Config.max_sentences)
                valid_claim_features, valid_evidence_features = load_feature_by_data_set(Config.dev_set_file,
                                                                                         Config.feature_path,
                                                                                         Config.max_sentences)
                training_set['data']['h_feats'] = training_claim_features
                training_set['data']['b_feats'] = training_evidence_features
                valid_set['data']['h_feats'] = valid_claim_features
                valid_set['data']['b_feats'] = valid_evidence_features
            if use_numeric_feature:
                training_num_feat = number_feature(Config.training_set_file, Config.db_path, Config.max_sentences)
                valid_num_feat = number_feature(Config.dev_set_file, Config.db_path, Config.max_sentences)
                training_set['data']['num_feat'] = training_num_feat
                valid_set['data']['num_feat'] = valid_num_feat
            if use_inter_evidence_comparison:
                training_concat_sent_indices, training_concat_sent_sizes = generate_concat_indices_for_inter_evidence(
                    training_set['data']['b_np'],
                    training_set['data']['b_sent_sizes'],
                    Config.max_sentence_size, Config.max_sentences)
                training_set['data']['b_concat_indices'] = training_concat_sent_indices
                training_set['data']['b_concat_sizes'] = training_concat_sent_sizes
                valid_concat_sent_indices, valid_concat_sent_sizes = generate_concat_indices_for_inter_evidence(
                    valid_set['data']['b_np'],
                    valid_set['data'][
                        'b_sent_sizes'],
                    Config.max_sentence_size,
                    Config.max_sentences)
                valid_set['data']['b_concat_indices'] = valid_concat_sent_indices
                valid_set['data']['b_concat_sizes'] = valid_concat_sent_sizes
            if use_claim_evidences_comparison:
                training_all_evidences_indices, training_all_evidences_sizes = generate_concat_indices_for_claim(
                    training_set['data']['b_np'], training_set['data']['b_sent_sizes'], Config.max_sentence_size,
                    Config.max_sentences)
                training_set['data']['b_concat_indices_for_h'] = training_all_evidences_indices
                training_set['data']['b_concat_sizes_for_h'] = training_all_evidences_sizes
                valid_all_evidences_indices, valid_all_evidences_sizes = generate_concat_indices_for_claim(
                    valid_set['data']['b_np'], valid_set['data']['b_sent_sizes'], Config.max_sentence_size,
                    Config.max_sentences)
                valid_set['data']['b_concat_indices_for_h'] = valid_all_evidences_indices
                valid_set['data']['b_concat_sizes_for_h'] = valid_all_evidences_sizes
            X_dict = {
                'X_train': training_set['data'],
                'X_valid': valid_set['data'],
                'y_valid': valid_set['label'],
                'embedding': embeddings
            }
            y_train = training_set['label']
            if hasattr(Config, 'training_dump'):
                with open(Config.training_dump, 'wb') as f:
                    pickle.dump((X_dict, y_train), f, protocol=pickle.HIGHEST_PROTOCOL)
        if estimator is None:
            estimator = get_estimator(Config.estimator_name, Config.ckpt_folder)
        estimator.fit(X_dict, y_train)
        save_model(estimator, Config.model_folder, Config.pickle_name, logger)
        
    elif mode == 'test':
        # testing mode
        restore_param_required = estimator is None
        if estimator is None:
            estimator = load_model(Config.model_folder, Config.pickle_name)
            if estimator is None:
                estimator = get_estimator(Config.estimator_name, Config.ckpt_folder)
        vocab, embeddings = load_whole_glove(Config.glove_path)
        vocab = vocab_map(vocab)
        test_set, _, _, _, _ = embed_data_set_with_glove_2(Config.test_set_file, Config.db_path, vocab_dict=vocab,
                                                           glove_embeddings=embeddings,
                                                           threshold_b_sent_num=Config.max_sentences,
                                                           threshold_b_sent_size=Config.max_sentence_size,
                                                           threshold_h_sent_size=Config.max_claim_size)
        h_sent_sizes = test_set['data']['h_sent_sizes']
        h_sizes = np.ones(len(h_sent_sizes), np.int32)
        test_set['data']['h_sent_sizes'] = np.expand_dims(h_sent_sizes, 1)
        test_set['data']['h_sizes'] = h_sizes
        test_set['data']['h_np'] = np.expand_dims(test_set['data']['h_np'], 1)
        if use_extra_features:
            assert hasattr(Config, 'feature_path'), "Config should has feature_path if Config.use_feature is True"
            test_claim_features, test_evidence_features = load_feature_by_data_set(Config.test_set_file,
                                                                                   Config.feature_path,
                                                                                   Config.max_sentences)
            test_set['data']['h_feats'] = test_claim_features
            test_set['data']['b_feats'] = test_evidence_features
        if use_numeric_feature:
            test_num_feat = number_feature(Config.test_set_file, Config.db_path, Config.max_sentences)
            test_set['data']['num_feat'] = test_num_feat
        x_dict = {
            'X_test': test_set['data'],
            'embedding': embeddings
        }
        if use_inter_evidence_comparison:
            test_concat_sent_indices, test_concat_sent_sizes = generate_concat_indices_for_inter_evidence(
                test_set['data']['b_np'],
                test_set['data']['b_sent_sizes'],
                Config.max_sentence_size,
                Config.max_sentences)
            test_set['data']['b_concat_indices'] = test_concat_sent_indices
            test_set['data']['b_concat_sizes'] = test_concat_sent_sizes
        if use_claim_evidences_comparison:
            test_all_evidences_indices, test_all_evidences_sizes = generate_concat_indices_for_claim(
                test_set['data']['b_np'], test_set['data']['b_sent_sizes'], Config.max_sentence_size,
                Config.max_sentences)
            test_set['data']['b_concat_indices_for_h'] = test_all_evidences_indices
            test_set['data']['b_concat_sizes_for_h'] = test_all_evidences_sizes
        predictions = estimator.predict(x_dict, restore_param_required)
        generate_submission(predictions, test_set['id'], Config.test_set_file, Config.submission_file)
        if 'label' in test_set:
            print_metrics(test_set['label'], predictions, logger)
    else:
        logger.error("Invalid argument --mode: " + mode + " Argument --mode should be either 'train’ or ’test’")
    return estimator


def entrance(mode, config, estimator=None):
    if config is not None:
        Config.load_config(config)
    if Config.estimator_name == 'esim':
        main_fasttext(mode, config, estimator)
    else:
        main(mode, config, estimator)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='\'train\' or \'test\'', required=True)
    parser.add_argument('--config', help='/path/to/config/file, in JSON format')
    args = parser.parse_args()
    entrance(args.mode, args.config)
