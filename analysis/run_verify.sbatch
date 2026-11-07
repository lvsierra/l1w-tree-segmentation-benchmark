#!/bin/bash
#SBATCH -J verify_fp
#SBATCH -p nodes              # Barkla CPU partition - change if yours is different
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o verify_fp_%j.out

export PATH="$HOME/.conda/envs/TreeLearn/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
D=/mnt/scratch/users/$USER/tree-seg/ForestFormer3D/data/ForAINetV2/forainetv2_instance_data

python verify_boundary_fp.py \
  --methods methods.tsv \
  --out verify_out \
  --xyz "$D/TreeLearn_L1W_test_vert.npy" \
  --xyz-labels "$D/TreeLearn_L1W_test_ins_label.npy"
