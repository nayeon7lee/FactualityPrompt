
# #################################################################
# #             Generation PPL (using MegatronLM)                 #
# #################################################################

'''
CANNOT provide this one -- because it requries OUR_GEN code. 
'''

# TASK="OUR_GEN"


# VOCAB_FILE= # path to gpt2-vocab.json file of Megatron-LM
# MERGE_FILE= # path to gpt2-merges.txt file of Megatron-LM
# CHECKPOINT_PATH= # path to GPT-345M checkpoint from Megatron-LM repository


# COMMON_TASK_ARGS="
#                   --num-layers 24 \
#                   --hidden-size 1024 \
#                   --num-attention-heads 16 \
#                   --seq-length 1024 \
#                   --max-position-embeddings 1024 \
#                   --fp16 \
#                   --vocab-file $VOCAB_FILE"


# # path to gen file to evaluate
# GEN_DIR=/gpfs/fs1/projects/gpu_adlr/outputs/nayeonl/generations
# VALID_DATA_NAME=$1

# # You can find this 
# python Megatron-LM/tasks/main.py \
#       --task $TASK \
#       $COMMON_TASK_ARGS \
#       --valid-data ${GEN_DIR}/${VALID_DATA_NAME} \
#       --tokenizer-type GPT2BPETokenizer \
#       --merge-file $MERGE_FILE \
#       --load $CHECKPOINT_PATH \
#       --micro-batch-size 16 \
#       --checkpoint-activations \
#       --log-interval 10 \
#       --no-load-optim \
#       --no-load-rng