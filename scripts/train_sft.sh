python train_sft.py \
  --model_path /mnt/sdb/ww/ckpt/qwen-7b \
  --train_data /home/wuwei/homework/llm/tuneqwen/final_high_quality_data/result_20250512_1109.jsonl \
  --output_dir ./sft_results \
  --epochs 3 \
  --batch_size 2 \
  --grad_accum 4 &> ./logs/sft.txt