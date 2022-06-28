#!/bin/bash


# bash script
python -m spacy download en_core_web_sm

# pip install -r requirements.txt


pip install fever-drqa

pip install hydra-core
# pip uninstall sacrebleu; pip install sacrebleu==1.5.1

pip install tensorflow
pip install torch==1.5.0
pip install torchvision==0.7.0


# for SentenceTransformer retriever
pip install torch==1.6.0
pip install -U sentence-transformers # (tokenizer==0.11.6, transformers==4.17.0)

# python in bash
python - << EOF
import nltk
import benepar
nltk.download('stopwords')
nltk.download('punkt')
benepar.download('benepar_en2')
EOF