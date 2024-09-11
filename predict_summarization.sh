#!/bin/bash
#SBATCH --account=ENT213036                 # parent project to access twcc system
#SBATCH --job-name=summarization            # jobName
#SBATCH --nodes=1                           # request node number
#SBATCH --ntasks-per-node=1                 # number of tasks can execute on the node
#SBATCH --gpus-per-node=1                   # gpus per node
#SBATCH --cpus-per-task=4                   # cpus per gpu
#SBATCH --partition=gp1d                    # how long task can run
#SBATCH --output=./summarization.%j.out     # specify output Diectory and fileName


ml miniconda3
conda activate ASA_classification

CUDA_VISIBLE_DEVICES=0 python summarization_LlaMA.py            \
--huggingface_token     yourtoken                               \
--model_name_or_path    meta-llama/Meta-Llama-3-8B              \
--test_file             ./test_data-10.json                     \
--output_dir            ./summarization_LlaMA3.json
