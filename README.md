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
* `metrics`: contains generation evaluation metrics used nucleus sampling [github](https://github.com/ari-holtzman/degen) from Holtman et al. 
* `prompts`: contains our FactualityPrompt testset utilized in our paper.
* `src`: our code that leverages the above resources to evaluate the factualtiy of LM generation.

## 1. Setup 
1. Install dependencies by running `pip install -r requirements.txt`
2. Download Wikipedia processed dump (knowledgesource.json) from [here](https://github.com/facebookresearch/KILT#kilt-knowledge-source) into `data` directory
3. Create the DB file from Wikipedia dump by running:

```
  PYTHONPATH=fever_athene python3 fever_athene/scripts/build_db_kilt.py data/knowledgesource.json data/kilt_db.db
```
