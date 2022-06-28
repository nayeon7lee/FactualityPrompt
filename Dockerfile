FROM nvcr.io/nvidia/pytorch:20.12-py3
# TODO: need to update this starting docker to something public!

RUN apt-get update && apt-get install  -y python pip

RUN pip install thefuzz
RUN pip install spacy
RUN pip install bitarray
RUN pip install datasets
RUN pip install sentence-transformers==2.2.0
RUN pip install Cython==0.29.15
RUN pip install numpy==1.19.1
RUN pip install benepar==0.1.3
# RUN pip install torch==1.5.0
RUN pip install fairseq==0.9.0
RUN pip install nltk==3.5
RUN pip install spacy==2.3.2
# RUN pip install tensorflow==1.15.0
RUN pip install transformers==3.4.0p
RUN pip install tensorflow


# bash script
RUN python -m spacy download en_core_web_sm

# python in bash
RUN python - << EOF
import nltk
import benepar
nltk.download('stopwords')
nltk.download('punkt')
benepar.download('benepar_en2')
EOF