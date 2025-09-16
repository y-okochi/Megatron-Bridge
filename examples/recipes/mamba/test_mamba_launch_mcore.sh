#!/bin/bash

#SBATCH --nodes=92
#SBATCH -t 12:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --dependency=singleton
#SBATCH --job-name=12B_hybrid_20T_phase1

export UB_TIMEOUT=720
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NCCL_P2P_NET_CHUNKSIZE=2097152
export NCCL_DEBUG=WARN
export TORCHINDUCTOR_WORKER_START=fork

CURRENT_DIR=`pwd`
OUTPUT_ROOT=/tmp #"${CURRENT_DIR}/12b_hybrid"
IMAGE_PATH="/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/nemotron5p5/images/adlr+megatron-lm+pytorch+nemotron5p5-apr2025-nvrx-patchedte.sqsh"
########################################################
#### CHANGES SHOULD NOT BE NEEDED BEYOND THIS POINT ####
########################################################

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
NAME="12B_hybrid_20T_phase1"

if [ -n "${SLURM_JOB_ID:-}" ] ; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
    ENV_LOG_FILENAME=${NAME}_${SLURM_JOB_ID}_${DATETIME}.env.log
else
    SCRIPT_PATH=$(realpath "$0")
    ENV_LOG_FILENAME=${NAME}_${DATETIME}.env.log
fi

SCRIPT_DIR=$(dirname ${SCRIPT_PATH})
REPO_DIR=/opt/megatron-lm #${CURRENT_DIR}

RUN_DIR="${OUTPUT_ROOT}"
LOGS_DIR="${RUN_DIR}/logs/phase1"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints/phase1"
DATACACHE_DIR="${RUN_DIR}/../data_cache"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard/phase1"

# Mamba triton cache.
# export TRITON_CACHE_DIR="/home/dnarayanan/nemotron5p5/triton_cache"
# export TRITON_CACHE_MANAGER="megatron.core.ssm.triton_cache_manager:ParallelFileCacheManager"

mkdir -p ${LOGS_DIR}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${DATACACHE_DIR}
mkdir -p ${TENSORBOARD_DIR}
################################################################
### Log environment
################################################################
echo "<< START PATHS >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "IMAGE_PATH=${IMAGE_PATH}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "OUTPUT_ROOT=${OUTPUT_ROOT}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "SCRIPT_DIR=${SCRIPT_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "REPO_DIR=${REPO_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "RUN_DIR=${RUN_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "LOGS_DIR=${LOGS_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "DATACACHE_DIR=${DATACACHE_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "TENSORBOARD_DIR=${TENSORBOARD_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "SCRIPT_DIR=${SCRIPT_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "REPO_DIR=${REPO_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END PATHS >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

echo "<< START GIT >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "GIT LOG" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
git -C ${REPO_DIR} log --oneline -1 |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "GIT STATUS" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
git -C ${REPO_DIR} status --porcelain --branch |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "GIT DIFF" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
git -C ${REPO_DIR} diff |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END GIT >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

echo "<< START ENV >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
env |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END ENV >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

# Tokenizer model.
TOKENIZER_MODEL="/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/nemotron5p5/tokenizers/multiMixV8.gpt4o_nc_sd.500000.128k.vocab.json"

# Data blend.
BLEND_PATH="/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/nemotron5p5/blend_files/20t_phase1.json"

# Optional args that can be incorporated later.
# --async-save
# --tp-comm-overlap (can't be used with fp8)

# --fp8-recipe blockwise \
#     --fp8-format e4m3 \
#     --first-last-layers-bf16 \
#     --num-layers-at-start-in-bf16 2 \
#     --num-layers-at-end-in-bf16 2 \
#     --fp8-param-gather \
    # --hybrid-override-pattern M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M- \
    # --per-split-data-args-path ${BLEND_PATH} \

options=" \
    --attention-backend flash \
    --disable-gloo-process-groups \
    --is-hybrid-model \
    --mamba-head-dim 80 \
    --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
    --mock-data \
    --tiktoken-pattern v2 \
    --distributed-timeout-minutes 120 \
    --use-mcore-models \
    --data-cache-path ${DATACACHE_DIR} \
    --no-mmap-bin-files \
    --hybrid-override-pattern M \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --init-method-std 0.0125 \
    --position-embedding-type none \
    --squared-relu \
    --num-layers 1 \
    --hidden-size 5120 \
    --num-attention-heads 40 \
    --group-query-attention \
    --num-query-groups 8 \
    --ffn-hidden-size 20480 \
    --kv-channels 128 \
    --normalization RMSNorm \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --exit-duration-in-mins 710 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --train-samples 128 \
    --lr-decay-samples 100 \
    --lr-warmup-samples 28 \
    --lr 4.5e-4 \
    --min-lr 4.5e-6 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --log-interval 100 \
    --eval-iters 10 \
    --eval-interval 1000 \
    --tokenizer-type TikTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --load ${CHECKPOINT_DIR} \
    --save ${CHECKPOINT_DIR} \
    --save-interval 100 \
    --save-retain-interval 10000 \
    --ckpt-format torch_dist \
    --ckpt-fully-parallel-save \
    --ckpt-fully-parallel-load \
    --ckpt-assume-constant-structure \
    --log-progress  \
    --timing-log-option minmax \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --log-throughput \
    --log-energy \
    --bf16 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --use-distributed-optimizer \
    --ddp-num-buckets 5 \
    --no-create-attention-mask-in-dataloader \
    --manual-gc \
    --num-workers 1 \
    --log-straggler \
    --disable-straggler-on-startup \
    --straggler-minmax-count 16 \
    --check-weight-hash-across-dp-replicas-interval 20000 \
    --rerun-mode validate_results \
    --tensorboard-dir ${TENSORBOARD_DIR}"

# run_cmd="numactl --cpunodebind=$((SLURM_LOCALID/4)) --membind=$((SLURM_LOCALID/4)) python -u ${REPO_DIR}/pretrain_mamba.py ${options}"
torchrun --nproc_per_node 8 ${REPO_DIR}/pretrain_mamba.py ${options}

# srun -l \
#     --container-image "${IMAGE_PATH}" \
#     --container-mounts "/home:/home,/lustre:/lustre" \
#     --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
#     sh -c "${run_cmd}"