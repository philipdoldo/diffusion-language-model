#!/bin/bash

#SBATCH --job-name=t
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH -o /mnt/data_r60_1/adv_robust_project/diffusion-language-model/logs/train_%A.log
#SBATCH -e /mnt/data_r60_1/adv_robust_project/diffusion-language-model/logs/train_%A.err
#SBATCH --time=00-05:00:00

###source /home/pdoldo/fs/bin/activate
source $(conda info --base)/etc/profile.d/conda.sh
conda activate drose


nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
export TORCH_MAX_MEMORY_FRACTION=0.98

srun torchrun \
--nnodes $SLURM_NNODES \
--nproc_per_node 4 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29500 \
/mnt/data_r60_1/adv_robust_project/diffusion-language-model/train.py \
--config "/mnt/data_r60_1/adv_robust_project/diffusion-language-model/template.yaml"
