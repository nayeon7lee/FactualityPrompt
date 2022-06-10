FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN apt-get update
RUN apt-get install -y --no-install-recommends --allow-unauthenticated \
    zip \
    gzip \
    make \
    automake \
    gcc \
    build-essential \
    g++ \
    cpp \
    libc6-dev \
    man-db \
    autoconf \
    pkg-config \
    unzip \
    libffi-dev \
    software-properties-common \ 
    wget \ 
    git

ENV HOME "/root"

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda && rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH "$PATH:$HOME/miniconda/bin"
ENV LANG C.UTF-8

RUN mkdir /fever
WORKDIR /fever
RUN mkdir -p data/fever
RUN mkdir -p data/fasttext

RUN wget -nv https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip && unzip wiki.en.zip -d data/fasttext && rm wiki.en.zip
RUN wget https://public.ukp.informatik.tu-darmstadt.de/fever-2018-team-athene/claim_verification_esim.ckpt.zip
RUN wget https://public.ukp.informatik.tu-darmstadt.de/fever-2018-team-athene/sentence_retrieval_ensemble.ckpt.zip
RUN wget https://public.ukp.informatik.tu-darmstadt.de/fever-2018-team-athene/document_retrieval_datasets.zip
RUN wget https://public.ukp.informatik.tu-darmstadt.de/fever-2018-team-athene/claim_verification_esim_glove_fasttext.ckpt.zip

RUN mkdir -p model/no_attention_glove/rte_checkpoints/
RUN mkdir -p model/esim_0/rte_checkpoints/
RUN mkdir -p model/esim_0/sentence_retrieval_ensemble/
RUN unzip claim_verification_esim.ckpt.zip -d model/no_attention_glove/rte_checkpoints/
RUN unzip claim_verification_esim_glove_fasttext.ckpt.zip -d model/esim_0/rte_checkpoints/
RUN unzip sentence_retrieval_ensemble.ckpt.zip -d model/esim_0/sentence_retrieval_ensemble/
RUN unzip document_retrieval_datasets.zip -d data/fever/

RUN wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
RUN unzip glove.6B.zip -d data/glove && rm glove.6B.zip
RUN gzip data/glove/*.txt

RUN rm *.zip

RUN conda install python=3.6
RUN conda install Cython=0.28.5
ADD requirements.txt /fever/
RUN pip install -r requirements.txt
RUN conda uninstall -y Cython
RUN pip uninstall -y pyfasttext
RUN pip install --force --upgrade cysignals==1.7.2
RUN pip install --force --upgrade pyfasttext
RUN conda install tensorflow=1.9.0 tensorflow-gpu=1.9.0

RUN python -c "import nltk; nltk.download('punkt')"

ADD src src
ADD server.sh .
ADD predict.sh .
ENV PYTHONPATH /fever/src
ENV PYTHONUNBUFFERED 1
CMD bash
#CMD python -m athene.system --db-path /local/fever-common/data/fever/fever.db --words-cache model/sentence --sentence-model model/esim_0/sentence_retrieval_ensemble
#CMD python src/rename.py --checkpoint_dir=model/esim_0/sentence_retrieval_ensemble/model1 --add_prefix=model_0/
CMD ["waitress-serve", "--host=0.0.0.0","--port=5000", "--call", "athene.system:web"]
#CMD ["bash", "./server.sh"]