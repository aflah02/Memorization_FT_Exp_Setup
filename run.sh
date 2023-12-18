#!/bin/bash

export HF_HOME=/NS/llm-1/nobackup/afkhan/HF_CACHE/Misc
export HF_DATASETS_CACHE=/NS/llm-1/nobackup/afkhan/HF_CACHE/Datasets
export TRANSFORMERS_CACHE=/NS/llm-1/nobackup/afkhan/HF_CACHE/Models

# Define the fine-tuning methods to explore
# fine_tuning_methods=("full-finetuning" "prefix-tuning" "prompt-tuning" "lora" "ia3" "p-tuning" "freeze-subset")
fine_tuning_methods=("full-finetuning" "prefix-tuning" "prompt-tuning" "lora" "ia3" "p-tuning")

# Define the pairs of values for seq_len and num_seq
# seq_len_values=(1024 512 256 128 64 32 16)
# num_seq_values=(1 2 4 8 16 32 64)
seq_len_values=(1024 512)
num_seq_values=(1 1)

# Iterate over each fine_tuning_method
for fine_tuning_method in "${fine_tuning_methods[@]}"
do
    # Iterate over pairs of values for seq_len and num_seq
    for ((i=0; i<${#seq_len_values[@]}; i++))
    do
        seq_len="${seq_len_values[$i]}"
        num_seq="${num_seq_values[$i]}"

        echo "Running with fine_tuning_method: $fine_tuning_method, seq_len: $seq_len, num_seq: $num_seq"
        python your_script.py \
            --fine_tuning_method "$fine_tuning_method" \
            --seq_len "$seq_len" \
            --num_seq "$num_seq"
    done
done