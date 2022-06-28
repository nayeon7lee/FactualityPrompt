#!/bin/bash

FT_CHECKPOINT_NAME=TODO

bash setup.sh

for PROMPT_TYPE in factual nonfactual
do
    DECODING_CHOICE=0.9 
    GEN_TO_EVALUATE_NAME=${PROMPT_TYPE}-${FT_CHECKPOINT_NAME}-${DECODING_CHOICE}_1234.jsonl
    # 0. factuality measure
    PYTHONPATH=. python src/evaluate_v3_final.py --prompt_type ${PROMPT_TYPE} --gen_path ${GEN_TO_EVALUATE_NAME} 
    # 1. repetition
    python metrics/repetition.py ${GEN_TO_EVALUATE_NAME}  --final
    # 2. PPL
    bash /root/megatron-lm/sh/evaluate_ppl_final.sh ${GEN_TO_EVALUATE_NAME}
   
    # 3. Diversity - TODO: later
    DECODING_CHOICE="${DECODING_CHOICE_LIST[$i]}"
    
    echo "$i" "|" $DECODING_CHOICE "|" $DECODING_ARGS


    model_size=1.3b
    # GEN_DIR=/gpfs/fs1/projects/gpu_adlr/outputs/nayeonl/generations/t1_diversity
    GEN_DIR=/gpfs/fs1/projects/gpu_adlr/outputs/nayeonl/generations/rebuttal_t1_diversity

    FILE_TEMPLATE=${model_size}-vanilla-${DECODING_CHOICE}_

    python metrics/distinct_n.py --gen_dir ${GEN_DIR} --file_template ${FILE_TEMPLATE}

done
