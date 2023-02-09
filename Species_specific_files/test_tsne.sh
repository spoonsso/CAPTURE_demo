#!/bin/bash
#SBATCH --job-name=jhwTest
#SBATCH --mem=128000
#SBATCH -t 6-23:59
#SBATCH -N 1
#SBATCH -n 12
#SBATCH -p tdunn
#SBATCH --gres=gpu:3

source activate capture5
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/hpc/group/tdunn/joshwu/miniconda3/envs/capture1/lib/
python tsne_test.py
