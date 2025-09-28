# First stage of LLM2Rec training -- Collaborative Supervised Fine-Tuning (CSFT).

model_path="/home/$USER/huggingface_data/hub/Qwen2-0.5B"

for category in "AmazonMix-6"
do
    train_file=$(ls -f ./data/${category}/5-core/train/${category}*.csv)
    eval_file=$(ls -f ./data/${category}/5-core/valid/${category}*.csv)
    echo ${train_file} ${info_file}

    # Change from source code: Only 1 cpu node and 1 process per node

    CUDA_VISIBLE_DEVICES=0 torchrun --master_port=25649 --nproc_per_node 1 \
        ./llm2rec/run_csft.py \
        --base_model ${model_path} \
        --train_file ${train_file} \
        --eval_file ${eval_file} \
        --output_dir ./output/Qwen2-0.5B-CSFT-${category} \
        --wandb_run_name Qwen2-0.5B-CSFT-${category} \
        --category ${category} \
        --train_from_scratch False \
        --use_lora False

    cp ${model_path}/*token* ./output/Qwen2-0.5B-CSFT-${category}/
    
    # Also copy tokenizer to the last checkpoint
    latest_ckpt=$(ls -d ./output/Qwen2-0.5B-CSFT-${category}/checkpoint-* | sort -V | tail -n 1)
    cp ${model_path}/*token* ${latest_ckpt}/
done