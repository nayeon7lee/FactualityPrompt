import json
import os
import os.path as path


class Config:

    @classmethod
    def load_config(cls, conf_path):
        with open(conf_path) as f:
            conf = json.load(f)
            for k, v in conf.items():
                setattr(cls, k, v)
            cls.make_all_dirs()

    @classmethod
    def save_config(cls, conf_path):
        obj = {}
        for k, v in cls.__dict__.items():
            if not isinstance(v, classmethod) and not k.startswith('__'):
                obj.update({k: v})
        with open(conf_path, 'w') as f:
            json.dump(obj, f, indent=4)

    @classmethod
    def make_all_dirs(cls):
        os.makedirs(cls.model_folder, exist_ok=True)
        os.makedirs(cls.ckpt_folder, exist_ok=True)
        os.makedirs(cls.submission_folder, exist_ok=True)
        # os.makedirs(cls.sentence_retrieval_model_folder, exist_ok=True)
        # os.makedirs(cls.sentence_retrieval_embedding_folder, exist_ok=True)
        os.makedirs(cls.sentence_retrieval_ensemble_param['model_path'], exist_ok=True)

    DATA_DIR='/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/checkpoints/fever-athene'

    SUBMISSION_FILE_NAME = "predictions.jsonl"
    model_name = "esim_0"
    glove_path = "{}/data/glove/glove.6B.300d.txt.gz".format(DATA_DIR)
    fasttext_path = "{}/data/fasttext/wiki.en.bin".format(DATA_DIR)
    # fasttext_path = path.join(BASE_DIR, "data/fasttext/fasttext.p")
    model_folder = "{}/model/{}".format(DATA_DIR, model_name)
    ckpt_folder = path.join(model_folder, 'rte_checkpoints')
    db_path =  "/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/db/fever.db"
    dataset_folder = "{}/data/fever".format(DATA_DIR)
    
    raw_training_set = "/local/fever-common/data/fever-data/train.jsonl" # CHECK
    raw_dev_set = "/local/fever-common/data/fever-data/dev.jsonl"
    raw_test_set = "/local/fever-common/data/fever-data/test.jsonl"
    
    training_doc_file = path.join(dataset_folder, "train.wiki7.jsonl")
    dev_doc_file = path.join(dataset_folder, "dev.wiki7.jsonl")
    test_doc_file = path.join(dataset_folder, "test.wiki7.jsonl")

    training_set_file = path.join(dataset_folder, "train.p7.s5.jsonl") # CHECK
    dev_set_file = path.join(dataset_folder, "dev.p7.s5.jsonl")
    test_set_file = path.join(dataset_folder, "test.p7.s5.jsonl")
    
    document_k_wiki = 7
    document_parallel = True
    document_add_claim = True
    # sentence_retrieval_model_name = "esim"
    # sentence_retrieval_model_folder = path.join(model_folder, "sentence_retrieval")
    # sentence_retrieval_embedding_folder = path.join(dataset_folder, "sentence_retrieval_embedding")
    submission_folder = path.join("data/submission")
    submission_file = path.join(submission_folder, SUBMISSION_FILE_NAME)
    estimator_name = "esim"
    pickle_name = estimator_name + ".p"
    esim_hyper_param = {
        # 'num_neurons': [
        #     250,
        #     180,
        #     900,
        #     550,
        #     180
        # ],
        'num_neurons': [
            250,
            180,
            180,
            900,
            550
        ],
        'lr': 0.002,
        'dropout': 0,
        'batch_size': 8,
        'pos_weight': [0.408658712, 1.942468514, 1.540587559],
        'max_checks_no_progress': 10,
        'trainable': False,
        'lstm_layers': 1,
        'optimizer': 'adam',
        'num_epoch': 100,
        'activation': 'relu',
        'initializer': 'he'
    }
    max_sentences = 5
    max_sentence_size = 50
    max_claim_size = max_sentence_size
    # n_jobs_ensemble = 2
    # seed = [55, 42, 666, 1234, 4321]
    seed = 55
    # vocab_file = path.join(BASE_DIR, 'vocab.p')
    name = 'claim_verification_esim'
    sentence_retrieval_ensemble_param = {
        'num_model': 2,
        'random_seed': 1234,
        'tf_random_state': [88, 12345, 4444, 8888, 9999],
        'num_negatives': 5,
        'c_max_length': 20,
        's_max_length': 60,
        'reserve_embed': False,
        'learning_rate': 0.001,
        'batch_size': 8,
        'num_epoch': 20,
        'dropout_rate': 0.1,
        'num_lstm_units': 128,
        'share_parameters': False,
        'model_path': path.join(model_folder, 'sentence_retrieval_ensemble')
    }
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(ckpt_folder, exist_ok=True)
    os.makedirs(submission_folder, exist_ok=True)
    # os.makedirs(sentence_retrieval_model_folder, exist_ok=True)
    # os.makedirs(sentence_retrieval_embedding_folder, exist_ok=True)
    os.makedirs(sentence_retrieval_ensemble_param['model_path'], exist_ok=True)
