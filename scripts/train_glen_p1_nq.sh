USE_DDP=false

if [ $USE_DDP = false ]; then
    # Without distributed training
    CUDA_VISIBLE_DEVICES=1 \
    python examples/glen_phase1/train_glen.py \
        --output_dir logs/model_glen_nq/GLEN_P1_base \
        --model_name_or_path t5-base \
        --load_best_model_at_end True \
        --query_type gtq_doc_aug_qg \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --dropout_rate 0.1 \
        --Rdrop 0.15 \
        --aug_query True \
        --aug_query_type corrupted_query \
        --input_dropout 1 \
        --id_class t5_bm25_truncate_3 \
        --dataset_name nq320k \
        --test100 0 \
        --tree 1 \
        --pretrain_decoder True \
        --max_input_length 156 \
        --val_check_interval 0.1 \
        --tie_word_embeddings True \
        --decoder_input doc_rep \
        --max_output_length 5 \
        --num_return_sequences 10 \
        --logging_steps 100 \
        --overwrite_output_dir \
        --wandb_tag glen_base \
        --do_eval \
        --seed 42 
else 
    # With distributed training
    CUDA_VISIBLE_DEVICES=0,1 \
    python -m torch.distributed.launch --nproc_per_node=2 examples/glen_phase1/train_glen.py \
        --ddp_find_unused_parameters False \
        --output_dir logs/model_glen_nq/GLEN_base \
        --model_name_or_path t5-base \
        --load_best_model_at_end True \
        --query_type gtq_doc_aug_qg \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --dropout_rate 0.1 \
        --Rdrop 0.15 \
        --aug_query True \
        --aug_query_type corrupted_query \
        --input_dropout 1 \
        --id_class t5_bm25_truncate_3 \
        --dataset_name nq320k \
        --test100 0 \
        --tree 1 \
        --pretrain_decoder True \
        --max_input_length 156 \
        --val_check_interval 0.1 \
        --tie_word_embeddings True \
        --decoder_input doc_rep \
        --max_output_length 5 \
        --num_return_sequences 10 \
        --logging_steps 100 \
        --overwrite_output_dir \
        --wandb_tag glen_base \
        --do_eval \
        --seed 42 
fi
