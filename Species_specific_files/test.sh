#!/bin/bash
#SBATCH --job-name=jhwTest
#SBATCH --mem=200000
#SBATCH -t 6-23:59
#SBATCH -N 1
#SBATCH -n 12
#SBATCH -p tdunn
#SBATCH --gres=gpu:1

source activate capture
module load Matlab/R2020b
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/hpc/group/tdunn/joshwu/miniconda3/envs/capture/lib/
# matlab -batch "clear;predsfile='truncated_preds.mat';pd_analysis_demo"
# matlab -batch "clear;predsfile='predictions.mat';pd_analysis_demo"
matlab -batch "clear;predsfile='duped_preds.mat';pd_analysis_demo"