# With distributed training
# Currently we do not allow distributed training for phase 2 due to gradient cache.


# Without distributed training
CUDA_VISIBLE_DEVICES=0 python examples/glen_phase2/train_glen.py \
  --output_dir logs/model_glen_nq/GLEN_P2_base \
  --load_pretrained_st5_checkpoint "checkpoint/glen_p1_nq/pytorch_model.bin" \
  --model_name_or_path t5-base \
  --save_steps 100 \
  --per_device_train_batch_size 128 \
  --positive_passage_no_shuffle \
  --learning_rate 5e-5 \
  --q_max_len 32 \
  --p_max_len 156 \
  --num_train_epochs 30 \
  --logging_steps 10 \
  --overwrite_output_dir \
  --softmax_temperature 1.0 \
  --query_type gtq \
  --train_n_passages 9 \
  --dataset_name nq320k \
  --num_multi_vectors 3 \
  --infonce_loss 1.0 \
  --q_to_docid_loss 0.5 \
  --cosine_point_loss 0.25 \
  --warmup_ratio 0.0 \
  --mask_special_tokens_for_decoding True \
  --do_docid_temperature_annealing True \
  --docid_temperature 1.0 \
  --docid_temperature_min 1e-5 \
  --negative_passage_type self \
  --grad_cache \
  --gc_q_chunk_size 128 \
  --gc_p_chunk_size 128