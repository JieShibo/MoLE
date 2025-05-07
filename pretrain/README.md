# Pretraining

We trained all models on the Ascend 910B NPU cluster using [MindSpeed-LLM](https://gitee.com/ascend/MindSpeed-LLM) framework. Here, we provide training code that can run on NVIDIA GPUs, based on [Megatron-LM core_r0.6.0](https://github.com/NVIDIA/Megatron-LM/tree/core_r0.6.0).

## Install

```bash
cd <path>/MoLE/pretrain
pip install -e .
```

## Data 

We train on the deduplicated Pile dataset. For data preprocessing, please refer to [Pythia](https://github.com/EleutherAI/pythia?tab=readme-ov-file#reproducing-training).

## Start Training
Here is an example training MoLE-4E-160M:
```bash
DISTRIBUTED_ARGS="
  --nproc_per_node $GPU_PER_NODE \
  --nnodes $NNODES \
  --node_rank $NODE_RANK \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
  --use-mcore-models \
  --disable-bias-linear \
  --seq-length 2048 \
  --max-position-embeddings 2048 \
  --num-layers 12 \
  --hidden-size 768 \
  --ffn-hidden-size 3072 \
  --num-attention-heads 12 \
  --init-method-std 0.01 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --normalization RMSNorm \
  --position-embedding-type rope \
  --untie-embeddings-and-output-weights \
  --no-masked-softmax-fusion \
  --use-flash-attn \
  --no-check-for-nan-in-loss-and-grad \
  --use-distributed-optimizer \
  --overlap-param-gather \
  --overlap-grad-reduce \
  --sequence-parallel \
  --tokenizer-type PretrainedFromHF \
  --lr-decay-style cosine \
  --weight-decay 0.1 \
  --lr-warmup-fraction 0.01 \
  --clip-grad 1.0 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --bf16 \
  --tensorboard-dir <your TENSORBOARD_DIR> \
  --tensorboard-log-interval 1 \
  --split 100,0,0 \
  --log-interval 1 \
  --eval-iters 0 \
  --distributed-backend nccl \
  --data-path <your DATA_PATH> \
  --save  <your SAVE_PATH> \
  --global-batch-size 1024 \
  --micro-batch-size 8 \
  --lr 6e-4 \
  --min-lr 6e-5 \
  --save-interval 500 \
  --tokenizer-name-or-path ./tokenizer \
  --train-iters 50000 \
  --num-experts 4
```
`$DISTRIBUTED_ARGS` should be configured based on specific hardware setup (e.g., number of nodes, GPUs per node, and network settings). Additionally, `--micro-batch-size` needs to be adjusted according to the available GPU memory.


## Note
Since the training code was ported from MindSpeed & NPUs, we have not yet run a full training workflow on GPUs, so there may still be some bugs to fix. For example, the current code only supports DP, and support for TP, PP, and CP is not yet guaranteed. We will continue to improve and update the code in the future.
