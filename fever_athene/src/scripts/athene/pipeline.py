import argparse
import os
from enum import Enum

from athene.retrieval.document.docment_retrieval import main as document_retrieval_main
from athene.retrieval.sentences.ensemble import entrance as sentence_retrieval_ensemble_entrance
# from athene.retrieval.sentences.sentence_retrieval import main as sentence_retrieval_main
from athene.utils.config import Config
from common.util.log_helper import LogHelper
from scripts.athene.rte import entrance as rte_main


class Mode(Enum):
    PIPELINE = 1  # Run the whole pipeline, training & predicting
    PIPELINE_NO_DOC_RETR = 2  # Skip the document retrieval sub-task. Training & predicting
    PIPELINE_RTE_ONLY = 3  # Run only the RTE sub-task. Training & predicting
    PREDICT = 4  # Run all 3 sub-tasks but no training, and only predict test set with pre-trained models of sentence retrieval and RTE.
    PREDICT_NO_DOC_RETR = 5  # Skip the document retrieval sub-task. No training and only predict test set with pre-trained models of sentence retrieval and RTE.
    PREDICT_RTE_ONLY = 6  # Predict test set only for the RTE sub-task with pre-trained model of RTE.
    PREDICT_ALL_DATASETS = 7  # Run all 3 sub-tasks but no training. Predict all 3 datasets for document retrieval and sentence retrieval.
    PREDICT_NO_DOC_RETR_ALL_DATASETS = 8  # Skip the document retrieval sub-task. No training. Predict all datasets for sentence retrieval.

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return Mode[s]
        except KeyError:
            raise ValueError()


# def sentence_retrieval_single_model(logger):
#     logger.info("Starting data pre-processing...")
#     tmp_file = os.path.join(Config.dataset_folder, "tmp.jsonl")
#     with open(tmp_file, 'w') as wf:
#         files = [Config.training_doc_file, Config.dev_doc_file, Config.test_doc_file]
#         for f in files:
#             with open(f) as rf:
#                 for line in rf:
#                     wf.write(line)
#     sentence_retrieval_main(model=Config.sentence_retrieval_model_name,
#                             model_store_dir=Config.sentence_retrieval_model_folder,
#                             training_set=Config.training_doc_file, dev_set=Config.dev_doc_file,
#                             test_set=tmp_file, output_test_set=None,
#                             embedding_path=Config.sentence_retrieval_embedding_folder, mode="data_preprocessing")
#     os.remove(tmp_file)
#     logger.info("Starting training sentence retrieval and selecting sentences for dev set...")
#     os.remove(os.path.join(Config.sentence_retrieval_embedding_folder, "test_data.p"))
#     os.remove(os.path.join(Config.sentence_retrieval_embedding_folder, "test_indexes.p"))
#     sentence_clf = sentence_retrieval_main(model=Config.sentence_retrieval_model_name,
#                                            model_store_dir=Config.sentence_retrieval_model_folder,
#                                            training_set=Config.training_doc_file, dev_set=Config.dev_doc_file,
#                                            test_set=Config.dev_doc_file, output_test_set=Config.dev_set_file,
#                                            embedding_path=Config.sentence_retrieval_embedding_folder)
#     logger.info("Finished training sentence retrieval and selecting sentences for test set.")
#     logger.info("Starting selecting sentences for training set...")
#     os.remove(os.path.join(Config.sentence_retrieval_embedding_folder, "test_data.p"))
#     os.remove(os.path.join(Config.sentence_retrieval_embedding_folder, "test_indexes.p"))
#     sentence_clf = sentence_retrieval_main(model=Config.sentence_retrieval_model_name,
#                                            model_store_dir=Config.sentence_retrieval_model_folder,
#                                            training_set=Config.training_doc_file, dev_set=Config.dev_doc_file,
#                                            test_set=Config.training_doc_file, output_test_set=Config.training_set_file,
#                                            embedding_path=Config.sentence_retrieval_embedding_folder, mode="test",
#                                            clf=sentence_clf)
#     logger.info("Finished selecting sentences for training set.")
#     logger.info("Starting selecting sentences for test set...")
#     os.remove(os.path.join(Config.sentence_retrieval_embedding_folder, "test_data.p"))
#     os.remove(os.path.join(Config.sentence_retrieval_embedding_folder, "test_indexes.p"))
#     sentence_clf = sentence_retrieval_main(model=Config.sentence_retrieval_model_name,
#                                            model_store_dir=Config.sentence_retrieval_model_folder,
#                                            training_set=Config.training_doc_file, dev_set=Config.dev_doc_file,
#                                            test_set=Config.test_doc_file, output_test_set=Config.test_set_file,
#                                            embedding_path=Config.sentence_retrieval_embedding_folder, mode="test",
#                                            clf=sentence_clf)
#     logger.info("Finished selecting sentences for dev set.")


def _construct_args_for_sentence_retrieval(phase='training'):
    from argparse import Namespace
    _args = Namespace()
    for k, v in Config.sentence_retrieval_ensemble_param.items():
        setattr(_args, k, v)
    setattr(_args, 'train_data', Config.training_doc_file)
    setattr(_args, 'dev_data', Config.dev_doc_file)
    setattr(_args, 'test_data', Config.test_doc_file)
    setattr(_args, 'fasttext_path', Config.fasttext_path)
    setattr(_args, 'phase', phase)
    if phase == 'training':
        out_file = Config.training_set_file
    elif phase == 'deving':
        out_file = Config.dev_set_file
    else:
        out_file = Config.test_set_file
    setattr(_args, 'out_file', out_file)
    return _args


def sentence_retrieval_ensemble(logger, mode: Mode = Mode.PIPELINE):
    logger.info("Starting data pre-processing...")
    tmp_file = os.path.join(Config.dataset_folder, "tmp.jsonl")
    with open(tmp_file, 'w') as wf:
        files = [Config.training_doc_file, Config.dev_doc_file, Config.test_doc_file]
        for f in files:
            with open(f) as rf:
                for line in rf:
                    wf.write(line)
    _args = _construct_args_for_sentence_retrieval()
    _args.phase = 'data'
    _args.test_data = tmp_file
    sentence_retrieval_ensemble_entrance(_args)
    os.remove(tmp_file)
    if mode in {Mode.PIPELINE, Mode.PIPELINE_NO_DOC_RETR}:
        logger.info("Starting training sentence retrieval...")
        _args.phase = 'training'
        _args.test_data = Config.dev_doc_file  # predict dev set in training phase
        os.remove(os.path.join(os.getcwd(), "test_data.p"))
        os.remove(os.path.join(os.getcwd(), "test_indexes.p"))
        sentence_retrieval_ensemble_entrance(_args)
        logger.info("Finished training sentence retrieval.")
    if mode in {Mode.PIPELINE, Mode.PIPELINE_NO_DOC_RETR, Mode.PREDICT_ALL_DATASETS,
                Mode.PREDICT_NO_DOC_RETR_ALL_DATASETS}:
        logger.info("Starting selecting sentences for dev set...")
        _args.phase = 'testing'
        _args.out_file = Config.dev_set_file
        _args.test_data = Config.dev_doc_file
        os.remove(os.path.join(os.getcwd(), "test_data.p"))
        os.remove(os.path.join(os.getcwd(), "test_indexes.p"))
        sentence_retrieval_ensemble_entrance(_args)
        logger.info("Finished selecting sentences for dev set.")
        logger.info("Starting selecting sentences for training set...")
        os.remove(os.path.join(os.getcwd(), "test_data.p"))
        os.remove(os.path.join(os.getcwd(), "test_indexes.p"))
        _args.test_data = Config.training_doc_file
        _args.phase = 'testing'
        _args.out_file = Config.training_set_file
        sentence_retrieval_ensemble_entrance(_args)
        logger.info("Finished selecting sentences for training set.")
    logger.info("Starting selecting sentences for test set...")
    os.remove(os.path.join(os.getcwd(), "test_data.p"))
    os.remove(os.path.join(os.getcwd(), "test_indexes.p"))
    _args.test_data = Config.test_doc_file
    _args.phase = 'testing'
    _args.out_file = Config.test_set_file
    sentence_retrieval_ensemble_entrance(_args, calculate_fever_score=False)
    logger.info("Finished selecting sentences for test set.")


def document_retrieval(logger, mode: Mode = Mode.PIPELINE):
    if mode in {Mode.PIPELINE, Mode.PREDICT_ALL_DATASETS}:
        logger.info("Starting document retrieval for training set...")
        document_retrieval_main(Config.db_path, Config.document_k_wiki, Config.raw_training_set,
                                Config.training_doc_file,
                                Config.document_add_claim, Config.document_parallel)
        logger.info("Finished document retrieval for training set.")
        logger.info("Starting document retrieval for dev set...")
        document_retrieval_main(Config.db_path, Config.document_k_wiki, Config.raw_dev_set, Config.dev_doc_file,
                                Config.document_add_claim, Config.document_parallel)
        logger.info("Finished document retrieval for dev set.")
    logger.info("Starting document retrieval for test set...")
    document_retrieval_main(Config.db_path, Config.document_k_wiki, Config.raw_test_set, Config.test_doc_file,
                            Config.document_add_claim, Config.document_parallel)
    logger.info("Finished document retrieval for test set.")


def rte(logger, mode: Mode = Mode.PIPELINE):
    claim_validation_estimator = None
    if mode in {Mode.PIPELINE_NO_DOC_RETR, Mode.PIPELINE, Mode.PIPELINE_RTE_ONLY}:
        logger.info("Starting training claim validation...")
        claim_validation_estimator = rte_main("train", args.config)
        logger.info("Finished training claim validation.")
    logger.info("Starting testing claim validation...")
    rte_main("test", args.config, claim_validation_estimator)
    logger.info("Finished testing claim validation.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='/path/to/config/file, in JSON format')
    parser.add_argument('--mode', type=Mode.from_string, choices=list(Mode), help='mode of the execution',
                        default=Mode.PIPELINE)
    args = parser.parse_args()
    LogHelper.setup()
    logger = LogHelper.get_logger(os.path.splitext(os.path.basename(__file__))[0])
    if args.config is not None:
        Config.load_config(args.config)
    if args.mode in {Mode.PIPELINE, Mode.PREDICT, Mode.PREDICT_ALL_DATASETS}:
        logger.info(
            "=========================== Sub-task 1. Document Retrieval ==========================================")
        document_retrieval(logger, args.mode)
    if args.mode in {Mode.PIPELINE_NO_DOC_RETR, Mode.PIPELINE, Mode.PREDICT, Mode.PREDICT_NO_DOC_RETR,
                     Mode.PREDICT_ALL_DATASETS, Mode.PREDICT_NO_DOC_RETR_ALL_DATASETS}:
        logger.info(
            "=========================== Sub-task 2. Sentence Retrieval ==========================================")
        sentence_retrieval_ensemble(logger, args.mode)
    logger.info("=========================== Sub-task 3. Claim Validation ============================================")
    rte(logger, args.mode)
