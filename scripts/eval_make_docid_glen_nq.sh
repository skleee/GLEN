NUM_MULTI_VECTORS=3
MAX_OUTPUT_LENGTH=$(($NUM_MULTI_VECTORS + 1))

# you can use --infer_dir or --infer_ckpt for other checkpoints
# e.g., --infer_dir logs/model_glen_nq/GLEN_P1_base/checkpoint-1234
# e.g., --infer_ckpt logs/model_glen_nq/GLEN_P1_base/checkpoint-1234/pytorch_model.bin

CUDA_VISIBLE_DEVICES=0 python examples/glen_phase2/makeid_glen.py \
  --model_name_or_path t5-base \
  --dataset_name nq320k \
  --max_input_length 156 \
  --per_device_eval_batch_size 16 \
  --id_class t5_bm25_truncate_3 \
  --docid_file_name glen_nq_docid \
  --max_output_length $MAX_OUTPUT_LENGTH \
  --num_return_sequences 10 \
  --infer_dir checkpoint/glen_p2_nq \
  --output_dir output
