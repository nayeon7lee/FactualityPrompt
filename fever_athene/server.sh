#!/usr/bin/env bash

python src/rename.py --checkpoint_dir=model/esim_0/sentence_retrieval_ensemble/model1 --add_prefix=model_0/
python src/rename.py --checkpoint_dir=model/esim_0/sentence_retrieval_ensemble/model2 --add_prefix=model_1/
python src/rename.py --checkpoint_dir=model/esim_0/sentence_retrieval_ensemble/model3 --add_prefix=model_2/
python src/rename.py --checkpoint_dir=model/esim_0/sentence_retrieval_ensemble/model4 --add_prefix=model_3/
python src/rename.py --checkpoint_dir=model/esim_0/sentence_retrieval_ensemble/model5 --add_prefix=model_4/

waitress-serve --host=0.0.0.0 --port=5000 --call athene.system:web