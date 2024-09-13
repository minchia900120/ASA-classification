#!/bin/bash
#SBATCH --account=ENT213036             # parent project to access twcc system
#SBATCH --job-name=train_lora           # jobName
#SBATCH --nodes=1                       # request node number
#SBATCH --ntasks-per-node=1             # number of tasks can execute on the node
#SBATCH --gpus-per-node=1               # gpus per node
#SBATCH --cpus-per-task=4               # cpus per gpu
#SBATCH --partition=gp1d                # how long task can run
#SBATCH --output=./train.%j.out         # specify output Diectory and fileName


ml miniconda3
conda activate ASA_classification

CUDA_VISIBLE_DEVICES=0 python   run_classification.py           \
--do_train                      True                            \
--do_eval                       True                            \
--model_name_or_path            meta-llama/Llama-2-7b-hf        \
--train_file                    train_data_label.json           \
--validation_file               validation_data_label.json      \
--text_column_names             instruction,input               \
--output_dir                    ./saves/LLaMA2-7B-hf/1epoch     \
--overwrite_cache               True                            \
--overwrite_output_dir          True                            \
--pad_to_max_length             False                           \
--max_seq_length                2048                            \
--preprocessing_num_workers     16                              \
--per_device_train_batch_size   1                               \
--per_device_eval_batch_size    1                               \
--gradient_accumulation_steps   8                               \
--lr_scheduler_type             cosine                          \
--logging_steps                 10                              \
--warmup_steps                  20                              \
--save_steps                    100                             \
--eval_steps                    100                             \
--evaluation_strategy           steps                           \
--load_best_model_at_end                                        \
--learning_rate                 5e-5                            \
--num_train_epochs              1.0                             \
--fp16                          True
