#!/bin/bash
#SBATCH --job-name=train_tissues
#SBATCH --output=logs/train_tissues_%j.out
#SBATCH --time=4-00:00:00
#SBATCH --mem=1400G
#SBATCH --gres=gpu:a40:8
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=2
#SBATCH --partition=gpu
#SBATCH --export=ALL

# Author: Jorge Ruiz-Orera
# Trains tissue + condition dual-FiLM RiboTransPred

# ============ ENVIRONMENT ============
source ~/.bashrc
mamba activate ribotranspred

# ============ NCCL (InfiniBand) ============
# Confirmed via `ip -o link show` + `ibstat` on maxg14/maxg23:
# - mlx5_0 is the real IB HCA used directly via RDMA (no IPoIB netdev exists,
#   so a name like ibp23s0 never resolves - that was the bootstrap failure).
# - bond176 is the active bonded NIC used for plain TCP traffic between nodes.
export NCCL_SOCKET_IFNAME=bond176
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_SL=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=SYS
export NCCL_BUFFSIZE=2097152
export NCCL_NTHREADS=64
export NCCL_SOCKET_NTHREADS=8
export NCCL_CHECKS_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=INIT,ENV
export NCCL_TIMEOUT=1800

export GLOO_SOCKET_IFNAME=bond176

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600

# ============ DISTRIBUTED ============
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((20000 + RANDOM % 45000))

# ============ DEBUG ============
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

# ============ TRAINING CONFIGURATION ============
TRACKS=$1
REGION_LEN=${2:-4500}
NBINS=$3
BIOTYPE="protein_coding"
MODEL=$4
SEED=$5

BATCH_SIZE=2
MAX_EPOCHS=80
DROPOUT=0.334
LR=2e-4
WEIGHT_DECAY=0.0037
WARMUP=2000
GRAD_ACCUM=3
GRAD_CLIP=0.5
TISSUE_EMB_DIM=32
COND_EMB_DIM=32

# ============ CONDITIONAL FLAGS ============
EXTRA_ARGS=""
if [ "$REGION_LEN" -eq "$NBINS" ]; then
    EXTRA_ARGS="--psites"
fi

# ============ DIAGNOSTICS ============
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "========================================="
echo "Node list: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_NNODES"
echo "Number of tasks: $SLURM_NTASKS"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "========================================="
echo "DDP Configuration:"
echo "Master node: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "NCCL_IB_DISABLE: $NCCL_IB_DISABLE"
echo "NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
echo "========================================="
echo "Training Configuration (Tissue + Condition dual FiLM):"
echo "REGION_LEN=$REGION_LEN"
echo "NBINS=$NBINS"
echo "BIOTYPE=$BIOTYPE"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "MAX_EPOCHS=$MAX_EPOCHS"
echo "DROPOUT=$DROPOUT"
echo "LR=$LR"
echo "WEIGHT_DECAY=$WEIGHT_DECAY"
echo "WARMUP=$WARMUP"
echo "GRAD_ACCUM=$GRAD_ACCUM"
echo "GRAD_CLIP=$GRAD_CLIP"
echo "TISSUE_EMB_DIM=$TISSUE_EMB_DIM"
echo "COND_EMB_DIM=$COND_EMB_DIM"
echo "Effective batch size: $((BATCH_SIZE * GRAD_ACCUM * SLURM_NTASKS))"
echo "Extra args: $EXTRA_ARGS"
echo "========================================="

mkdir -p results_tissues logs
echo "INPUT:standard"

# ============ SHARED SRUN ARGS ============
SRUN_ARGS="--wait=60 --kill-on-bad-exit=1 --cpu-bind=socket"

PYTHON_ARGS="
    --region_len     $REGION_LEN
    --nBins          $NBINS
    --tracks         $TRACKS
    --tracks_dir     tracks
    --model-type     $MODEL
    --save_path      results_tissues
    --biotype        $BIOTYPE
    --batch-size     $BATCH_SIZE
    --max-epochs     $MAX_EPOCHS
    --dropout        $DROPOUT
    --learning_rate  $LR
    --weight_decay   $WEIGHT_DECAY
    --warmup_steps   $WARMUP
    --grad_accum     $GRAD_ACCUM
    --grad_clip      $GRAD_CLIP
    --tissue_emb_dim $TISSUE_EMB_DIM
    --cond_emb_dim   $COND_EMB_DIM
    --num-workers    1
    --seed           $SEED
    $EXTRA_ARGS
"

# ============ TRAINING ============
echo "Starting dual FiLM training with srun..."
echo "========================================="

srun $SRUN_ARGS \
    python -u scripts/train_tissues.py \
        $PYTHON_ARGS \
    2>&1

TRAIN_EXIT=$?

if [ $TRAIN_EXIT -eq 0 ]; then
    echo "========================================="
    echo "Training completed successfully!"
    echo "========================================="

    # ============ TESTING ============
    echo "Testing with the validation-selected best.ckpt for this configuration"
    echo "========================================="

    srun $SRUN_ARGS \
        python -u scripts/train_tissues.py \
            $PYTHON_ARGS \
            --test \
        2>&1

    if [ $? -eq 0 ]; then
        echo "========================================="
        echo "Testing completed successfully!"
        echo "========================================="
    else
        echo "========================================="
        echo "Testing failed"
        echo "========================================="
        exit 1
    fi

else
    echo "========================================="
    echo "Training failed with exit code $TRAIN_EXIT"
    echo "========================================="
    exit 1
fi

echo "Job finished at: $(date)"
echo "========================================="
