#!/bin/bash
#SBATCH --job-name=sn_img_batch # (change)
#SBATCH --gres=gpu:p100:1 # number of gpus per node, p100l:16G (change)
#SBATCH --cpus-per-task=12
#SBATCH --nodes=1
#SBATCH --mem=32000M
#SBATCH --time=23:59:59 # time (DD-HH:MM)
#SBATCH --account=rrg-msavva # account
#SBATCH --output=%A_%a.out # %A for main jobID, %a for array id.
#SBATCH --error=%A_%a.err
#SBATCH --mail-user=yuer@sfu.ca
#SBATCH --mail-type=ALL

module load StdEnv/2020 python/3.7
source /home/yuer/projects/rrg-msavva/yuer/envs/py37-torch/bin/activate


python snare_train.py 
