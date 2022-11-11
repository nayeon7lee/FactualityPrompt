# FactualityPrompt
![License: Apache](https://img.shields.io/badge/License-Apache2.0-yellow.svg) 
  
This repository contains the test prompts and evaluation pipeline used in: "[Factuality Enhanced Language Models for
Open-Ended Text Generation](https://arxiv.org/pdf/2206.04624.pdf)". _Nayeon Lee, Wei Ping, Peng Xu, Mostofa Patwary, Pascale Fung, Mohammad Shoeybi, and Bryan Catanzaro_. 

This work was done during Nayeon Lee's internship at NVIDIA.

<!-- <img align="right" src="img/HKUST.jpg" width="12%"> -->

If you use our resource, please cite our work with the bibtex listed below:
```bibtex
@misc{https://doi.org/10.48550/arxiv.2206.04624,
  doi = {10.48550/ARXIV.2206.04624},
  url = {https://arxiv.org/abs/2206.04624},
  author = {Lee, Nayeon and Ping, Wei and Xu, Peng and Patwary, Mostofa and Fung, Pascale and Shoeybi, Mohammad and Catanzaro, Bryan},
  keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), Computers and Society (cs.CY), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Factuality Enhanced Language Models for Open-Ended Text Generation},
  publisher = {arXiv},
  year = {2022},  
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## Code Overview
* `fever_athene`: contains fact-checking pipeline code (Wiki document retriever, Wiki sentence selector, etc) from UKPLab/fever-2018-team-athene [github](UKPLab/fever-2018-team-athene). We utilize and build on top of their Wiki document retriever in our work. (Refer to their github for citation details)
* `prompts`: contains our FactualityPrompt testset utilized in our paper.
* `src`: codes for evaluating the factualtiy of LM generation (For files adapted from other publicly available codebases, we included the pointer to the original code file)

## 1. Setup 
1. Install dependencies by running `pip install -r requirements.txt`
2. Download Wikipedia processed dump (knowledgesource.json) from [KILT-github](https://github.com/facebookresearch/KILT#kilt-knowledge-source) into `data` directory (Refer to their repository for citation details)
```bash
  mkdir data
  cd data
  wget http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json
```
3. Create the DB file from Wikipedia dump by running:

```bash
  PYTHONPATH=fever_athene python3 fever_athene/scripts/build_db_kilt.py data/knowledgesource.json data/kilt_db.db
```
This script will create kilt_db.db into `data` directory. 

4. Configure `src/const.py` file. 

## 2. Run evaluation script
Running any of the scripts below will save corresponding metric results into a file named `$GEN_TO_EVALUATE_NAME_results.jsonl` (`$GEN_TO_EVALUATE_NAME` refers to the file containing generations that you are trying to evaluate).

### Factuality Metric (Hallucinated NE Error, Entailment Ratio)

```bash
for PROMPT_TYPE in factual nonfactual
do
    GEN_TO_EVALUATE_NAME=${PROMPT_TYPE}-CUSTOM-GEN-NAME.jsonl
    PYTHONPATH=. python src/evaluate_v3_final.py --prompt_type ${PROMPT_TYPE} --gen_path ${GEN_TO_EVALUATE_NAME}
done
```

### Repetition

```bash
for PROMPT_TYPE in factual nonfactual
do
    GEN_TO_EVALUATE_NAME=${PROMPT_TYPE}-CUSTOM-GEN-NAME.jsonl
    python src/repetition.py ${GEN_TO_EVALUATE_NAME}  --final
done
``` 

### Diversity

1. First obtain multiple generation files from your LM with different seed. In our paper, we used 10 random seeds, but you can use your own choice of seed count. **If you are evaluating greedy, there is NO NEED to generate multiple seed, because all seed will result in same generation. Simply use 1 generation file.**

2. Then run the below script:
```bash
GEN_DIR=directory-containing-multi-seed-generation-files

FILE_TEMPLATE=shared-string-between-multiple-seed-generation
python src/distinct_n.py --gen_dir ${GEN_DIR} --file_template ${FILE_TEMPLATE} --number_of_seeds 10
```

Illustration of `FILE_TEMPLATE`:
* Let's assume your generation files are named as follows: factual_gen_seed1.jsonl, nonfactual_gen_seed1.jsonl, factual_gen_seed2.jsonl, nonfactual_gen_seed2.jsonl,...
* Then, your `FILE_TEMPLATE` will be "gen_seed"


## 3. Replicating our work with [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) (Note: we used v3.0.2)
#### Factual Nucleus Decoding
Refer to this [link](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/text_generation/generation.py#L207) for nucleus sampling implementation in Megatron-LM repository.

#### Sentence Completition Loss
**Step 1. Prepare the training corpus:**

```python
  python preprocess_data_megatron_lm.py \
      --input $CORPUS_PATH \
      --output-prefix $OUTPUT_FILE_PREFIX \
      --vocab-file gpt2-vocab.json \
      --merge-file gpt2-merges.txt \
      --tokenizer-type GPT2BPETokenizer \
      --append-eod  --workers 20 \
      --mask_type $MASKING_CHOICE_FOR_SC_LOSS_PIVOT
```

Possible choice for `$MASKING_CHOICE_FOR_SC_LOSS_PIVOT`:
* `v2_all_after_ROOT`: ROOT Pivot 
* `v3_all_after_half`: Half Pivot
* `v5_RANDOM_Mask`: Random pivot

**Step 2: Modify the Megatron-LM code to incorporate SC-loss masking**

We just need to modify `get_batch()` function from <https://github.com/NVIDIA/Megatron-LM/blob/main/pretrain_gpt.py> file with below code snippet:

```python
def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    # keys = ['text'] # <- original code
    keys = ['text', 'ne_mask'] # <- our code
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    # '''Below three lines are our code'''
    if 'ne_mask' in keys:
        loss_mask_ = data_b['ne_mask'].long()
        loss_mask = loss_mask_[:, :-1].contiguous()
        
    return tokens, labels, loss_mask, attention_mask, position_ids
```

**Step 3: Use the provided script in Megtraon-LM repository (<https://github.com/NVIDIA/Megatron-LM#gpt-pretraining>) to continue training the Megatron-GPT with SC-loss**
* Set `DATA_PATH` to the preprocessed files generated from Step 1.
* Note that: since publicly available Megatron-LM checkpoint (345M) is smaller than the models used in our paper, same performance won't be replicated.

## 4. Replicating our work with [Hugginface](https://github.com/huggingface/transformers) (v4.20.1)
Please refer to this repository -> <https://github.com/nayeon7lee/factuality_enhanced_lm_hf>.
