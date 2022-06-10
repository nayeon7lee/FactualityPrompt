import pickle

import numpy as np


class CustomizedVotingClassifier:
    def __init__(self, prediction_path_list, voting):
        self.prediction_path_list = prediction_path_list
        self.voting = voting

    def fit(self, X, y):
        raise NotImplementedError(
            "This voting classifier is only used for combining existing models, not for training!")

    def _raw_probas(self):
        _probas = []
        for prediction_path in self.prediction_path_list:
            with open(prediction_path, 'rb') as f:
                _probas.append(pickle.load(f))
        return np.asarray(_probas)

    def predict_proba(self, X):
        # samples * classes
        _avg_probas = np.average(self._raw_probas(), axis=0)
        return np.argmax(_avg_probas, axis=1)

    def predict(self, X):
        if self.voting == 'soft':
            return self.predict_proba(X)
        else:
            # models * samples * classes
            _raw_probas = self._raw_probas()
            # models * samples
            _predictions_per_model = np.argmax(_raw_probas, axis=2)
            return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=_predictions_per_model.T)
