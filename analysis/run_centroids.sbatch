#!/bin/bash
#SBATCH -J fp_centroids
#SBATCH -p nodes
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o fp_centroids_%j.out

export PATH="$HOME/.conda/envs/TreeLearn/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
T=/mnt/scratch/users/$USER/tree-seg
D=$T/ForestFormer3D/data/ForAINetV2/forainetv2_instance_data

python export_centroids.py \
  --methods methods.tsv \
  --out verify_out \
  --xyz "$D/TreeLearn_L1W_test_vert.npy" \
  --gt-native "$T/output/gt/L1W_gt156_native85M.npy" \
  --sat-laz "$T/treelearn_results/L1W_SAT_out.laz" \
  --gt-sat "$T/output/gt/L1W_gt156_satAligned.npy"
