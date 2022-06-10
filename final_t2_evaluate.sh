#!/bin/bash

# # <!---- default ---->
FT1=gpt3-1.3b-bs-512-wiki-fever-default-2e7-500
# <!---- NEMask ---->
FT2=gpt3-1.3b-bs-512-wiki-fever-NEMask-2e6

# <!-- WikiNamePrefix-2e6 -->
FT3=gpt3-1.3b-bs-512-wiki-fever-WikiNamePrefix-2e6

# # <!-- v2ROOT + WikiNamePrefix-2e6 -->
FT4=gpt3-1.3b-bs-512-wiki-fever-WikiNamePrefix-v2ROOTMask-2e6

# # <!-- v3Half + WikiNamePrefix-2e6 -->
FT5=gpt3-1.3b-bs-512-wiki-fever-v3HALFMask-WikiNamePrefix-2e6

# # <!-- v3Half + WikiNamePrefix  -->
FT6=gpt3-1.3b-bs-512-wiki-fever-v3HALFMask-WikiNamePrefix-2e6

# # <!-- v4_ROOT_NEMask + WikiNamePrefix  -->
FT7=gpt3-1.3b-bs-512-wiki-fever-v4_ROOT_NEMask-WikiNamePrefix-2e6

# # <!-- v5_ROOT_NEMask + WikiNamePrefix  -->
FT8=gpt3-1.3b-bs-512-wiki-fever-v5_RANDOM_Mask-WikiNamePrefix-2e6


# # <!-- v2ROOT -->
FT9=gpt3-1.3b-bs-512-wiki-fever-v2ROOTMask-2e6

# # <!-- v3Half  -->
FT10=gpt3-1.3b-bs-512-wiki-fever-v3HALFMask-2e6

# # <!-- v4_ROOT_NEMask  -->
FT11=gpt3-1.3b-bs-512-wiki-fever-v4_ROOT_NEMask-2e6

# <!-- WikiNamePrefix-NEMask-2e6 -->
FT12=gpt3-1.3b-bs-512-wiki-fever-WikiNamePrefix-NEMask-2e6


CHECKPOINTS=($FT12) # $FT2 $FT3 $FT4 $FT5 $FT6 $FT7 $FT8 $FT9 $FT10 $FT11 $12)

## NOTE: I think i need to use tw_evaluate_debug.sh version

for FT_CHECKPOINT_NAME in "${CHECKPOINTS[@]}"
do
    cmd="
        bash /root/megatron-lm/setup.sh
        for PROMPT_TYPE in factual nonfactual
        do
            DECODING_CHOICE=0.9 
            GEN_TO_EVALUATE_NAME=${PROMPT_TYPE}-${FT_CHECKPOINT_NAME}-${DECODING_CHOICE}_1234.jsonl
            # 0. factuality measure
            PYTHONPATH=. python /root/megatron-lm/src/evaluate_v3_final.py --prompt_type ${PROMPT_TYPE} --gen_path ${GEN_TO_EVALUATE_NAME} 
            # 1. repetition
            python /root/megatron-lm/metrics/repetition.py ${GEN_TO_EVALUATE_NAME}  --final
            # 2. PPL
            bash /root/megatron-lm/sh/evaluate_ppl_final.sh ${GEN_TO_EVALUATE_NAME}
            # 3. Diversity - TODO: later

            # DECODING_CHOICE=0.9x0.9 
            # GEN_TO_EVALUATE_NAME=${PROMPT_TYPE}-${FT_CHECKPOINT_NAME}-${DECODING_CHOICE}_1234.jsonl
            # PYTHONPATH=. python /root/megatron-lm/src/evaluate_v3_final.py --prompt_type ${PROMPT_TYPE} --gen_path ${GEN_TO_EVALUATE_NAME}
        done
        "

    submit_job --image nvcr.io/nvidia/pytorch:20.12-py3 --mounts "/home/nayeonl,/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl,/gpfs/fs1/projects/gpu_adlr/outputs/nayeonl,/gpfs/fs1/projects/gpu_adlr/datasets/mpatwary/checkpoints/gpt3" --partition batch_dgx2h_m2 --gpu 1 --name factual_lm_eval --command "${cmd}"  --autoresume_timer 460    

done


# for FT_CHECKPOINT_NAME in "${CHECKPOINTS[@]}"
# do
#     for PROMPT_TYPE in factual nonfactual
#     do
#         # # <!-- P=0.9 decoding -->
#         DECODING_CHOICE=0.9
#         GEN_TO_EVALUATE_NAME=${PROMPT_TYPE}-${FT_CHECKPOINT_NAME}-${DECODING_CHOICE}
#         bash /root/megatron-lm/PYTHONPATH=. python src/evaluate_v3_final.py --prompt_type ${PROMPT_TYPE} --gen_path ${GEN_TO_EVALUATE_NAME}

#         # # <!-- P=0.9x0.9 -->
#         DECODING_CHOICE=0.9x0.9
#         GEN_TO_EVALUATE_NAME=${PROMPT_TYPE}-${FT_CHECKPOINT_NAME}-${DECODING_CHOICE}
#         bash /root/megatron-lm/PYTHONPATH=. python src/evaluate_v3_final.py --prompt_type ${PROMPT_TYPE} --gen_path ${GEN_TO_EVALUATE_NAME}
        
#         # # <!-- P=0.9x0.9 with lower_cap=0.3 -->
#         # TODO

#         # # <!-- P=0.9x0.9 with wiki-N-sent -->
#         DECODING_CHOICE=0.9x0.9
#         GEN_TO_EVALUATE_NAME=${PROMPT_TYPE}-${FT_CHECKPOINT_NAME}-${DECODING_CHOICE}-wiki1
#         bash /root/megatron-lm/PYTHONPATH=. python src/evaluate_v3_final.py --prompt_type ${PROMPT_TYPE} --gen_path ${GEN_TO_EVALUATE_NAME}
#     done
# done


# submit_job --image nvcr.io/nvidia/pytorch:20.12-py3 --mounts "/home/nayeonl,/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl,/gpfs/fs1/projects/gpu_adlr/outputs/nayeonl,/gpfs/fs1/projects/gpu_adlr/datasets/mpatwary/checkpoints/gpt3" --partition batch_dgx2h_m2 --gpu 1 --name factual_lm_eval_debug --command "bash /home/nayeonl/megatron-lm/final_t2_evaluate.sh"  --autoresume_timer 460

# factual_lm_eval_debug_20220507-010733, 391723