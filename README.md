# FactualityPrompt
![License: Apache](https://img.shields.io/badge/License-Apache2.0-yellow.svg) 
  
This repository contains the test prompts and evaluation pipeline used in: "[Factuality Enhanced Language Models for
Open-Ended Text Generation](https://arxiv.org/pdf/2206.04624.pdf)". _Nayeon Lee, Wei Ping, Peng Xu, Mostofa Patwary, Mohammad Shoeybi, and Bryan Catanzaro_. 

This work was done during Nayeon Lee's internship at NVIDIA.

<!-- <img align="right" src="img/HKUST.jpg" width="12%"> -->

If you use our resource, please cite our work with the bibtex listed below:
<pre>
@misc{https://doi.org/10.48550/arxiv.2206.04624,
  doi = {10.48550/ARXIV.2206.04624},
  url = {https://arxiv.org/abs/2206.04624},
  author = {Lee, Nayeon and Ping, Wei and Xu, Peng and Patwary, Mostofa and Shoeybi, Mohammad and Catanzaro, Bryan},
  keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), Computers and Society (cs.CY), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Factuality Enhanced Language Models for Open-Ended Text Generation},
  publisher = {arXiv},
  year = {2022},  
  copyright = {Creative Commons Attribution 4.0 International}
}
</pre>

## Code Overview
* `fever_athene`: contains fact-checking pipeline code (Wiki document retriever, Wiki sentence selector, etc) from UKPLab/fever-2018-team-athene [github](UKPLab/fever-2018-team-athene). We utilize and build on top of their Wiki document retriever in our work. (Refer to their github for citation details)
* `prompts`: contains our FactualityPrompt testset utilized in our paper.
* `src`: our code for evaluating the factualtiy of LM generation.

## 1. Setup 
1. Install dependencies by running `pip install -r requirements.txt`
2. Download Wikipedia processed dump (knowledgesource.json) from [KILT-github](https://github.com/facebookresearch/KILT#kilt-knowledge-source) into `data` directory (Refer to their repository for citation details)

```
  mkdir data
  cd data
  wget http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json
```
3. Create the DB file from Wikipedia dump by running:

```
  PYTHONPATH=fever_athene python3 fever_athene/scripts/build_db_kilt.py data/knowledgesource.json data/kilt_db.db
```
This script will create kilt_db.db into `data` directory. 

## 2. Run evaluation script
Running any of the scripts below will save corresponding metric results into a file named `$GEN_TO_EVALUATE_NAME_results.jsonl` (`$GEN_TO_EVALUATE_NAME` refers to the file containing generations that you are trying to evaluate).

#### Factuality Metric (Hallucinated NE Error, Entailment Ratio)

```
for PROMPT_TYPE in factual nonfactual
do
    GEN_TO_EVALUATE_NAME=${PROMPT_TYPE}-CUSTOM-GEN-NAME.jsonl
    PYTHONPATH=. python src/evaluate_v3_final.py --prompt_type ${PROMPT_TYPE} --gen_path ${GEN_TO_EVALUATE_NAME}
done
```

#### Repetition

```
for PROMPT_TYPE in factual nonfactual
do
    GEN_TO_EVALUATE_NAME=${PROMPT_TYPE}-CUSTOM-GEN-NAME.jsonl
    python src/repetition.py ${GEN_TO_EVALUATE_NAME}  --final
done
``` 

#### Diversity

1. First obtain multiple generation files from your LM with different seed. In our paper, we used 10 random seeds, but you can use your own choice of seed count.**If you are evaluating greedy, there is NO NEED to generate multiple seed, because all seed will result in same generation. Simply use 1 generation file.**

2. Then run the below script:
```
GEN_DIR=directory-containing-multi-seed-generation-files

FILE_TEMPLATE=shared-string-between-multiple-seed-generation
python src/distinct_n.py --gen_dir ${GEN_DIR} --file_template ${FILE_TEMPLATE} --number_of_seeds 10
```

Illustration of `FILE_TEMPLATE`:
* Let's assume your generation files are named as follows: factual_gen_seed1.jsonl, nonfactual_gen_seed1.jsonl, factual_gen_seed2.jsonl, nonfactual_gen_seed2.jsonl,...
* Then, your `FILE_TEMPLATE` will be "gen_seed"


## 3. Replicating our work with [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
1. Factual Nucleus Decoding: provide link to the implementation in Megatron-LM repo
2. How to 1) preprocess the training corpus, 2) incorporate SC-loss into Megatron-LM

## 4. Proposed methodology implementation in [Hugginface (HF)](https://github.com/huggingface/transformers)
1. Factual Nucleus Decoding with HF - https://github.com/huggingface/transformers/blob/main/examples/legacy/run_language_modeling.py 
2. How to incorporate SC-loss into HF codebase
