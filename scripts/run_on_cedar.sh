#!/bin/bash
#SBATCH --job-name=configX # (change, X is the setting's number)
#SBATCH --gres=gpu:p100:4 # number of gpus per node, p100l:16G (change)
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --time=1-05:59 # time (DD-HH:MM)
#SBATCH --account=rrg-msavva # account
#SBATCH --output=%A_%a.out # %A for main jobID, %a for array id.
#SBATCH --error=%A_%a.err
#SBATCH --mail-user=yuer@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0 # 0-3 means array of 4, each of them is one job submission with the above settings

module load StdEnv/2020 python/3.7
source /home/yuer/projects/rrg-msavva/yuer/envs/py37-torch/bin/activate


GPU=$((0))
JOB_ID=$((SLURM_ARRAY_TASK_ID))
pids=""
for GPU in 0 1 2 3
    do
        echo $JOB_ID $GPU
        SETTINGS_ID=$((JOB_ID*4+GPU))
        CUDA_VISIBLE_DEVICES=$GPU python run_retrieval.py --config_file t3ds/configs/retrieval_shapenet.yaml --expr_id v64i128b128  --gpu_id $GPU & 
        pids="$pids $!"
    done
wait $pids