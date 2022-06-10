import pytest

import numpy as np
from athene.retrieval.sentences.deep_models.ESIM import ESIM as ESIMretrieval
from athene.rte.deep_models.ESIM_for_ensemble_glove_only_no_attention import ESIM as ESIMrte
from athene.rte.utils.text_processing import load_whole_glove
from common.util.log_helper import LogHelper

LogHelper.setup()


def test_load_retrieval_model():
    dummy_embeddings = np.zeros((1, 300), dtype=np.float32)
    estimator = ESIMretrieval(
        h_max_length=20, s_max_length=60, learning_rate=0.001, batch_size=256, num_epoch=20,
        model_store_dir=None,
        embedding=dummy_embeddings,
        word_dict=None, dropout_rate=0.2, random_state=88, num_units=128,
        share_rnn=True
    )
    # estimator.restore_model("../models/retrieval/best_model.ckpt")
    estimator.restore_model("../models/retrieval/sentence_selection_esim.ckpt")


def test_load_rte_model():
    dummy_embeddings = np.zeros((1, 300), dtype=np.float32)
    estimator = ESIMrte(name='esim_verify',
                        activation='relu',
                        batch_size=64,
                        lstm_layers=1,
                        n_outputs=3,
                        num_neurons=[250, 180, 900, 550, 180],
                        show_progress=1, embedding=dummy_embeddings
                        )
    # estimator.restore_model("../models/rte/esim1.ckpt")
    estimator.restore_model("../models/rte/claim_verification_esim.ckpt")


@pytest.mark.skip(reason="Loading GloVe takes around 10 mins.")
def test_load_rte_model_2():
    vocab, embeddings = load_whole_glove("../../resources/embeddings/glove/glove.6B.300d.txt")
    estimator = ESIMrte(name='esim_verify',
                        activation='relu',
                        batch_size=64,
                        lstm_layers=1,
                        n_outputs=3,
                        num_neurons=[250, 180, 900, 550, 180],
                        show_progress=1, embedding=embeddings, vocab_size=len(vocab)
                        )
    estimator.restore_model("../models/rte/claim_verification_esim.ckpt")


if __name__ == "__main__":
    pytest.main([__file__])
