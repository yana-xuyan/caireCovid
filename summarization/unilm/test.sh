# TRAIN_FILE=/home/tiezheng/workspace/kaggle/unilm/s2s-ft/checkpoint/ckpt-22500
# SPLIT=validation
# INPUT_JSON= ../data/validation.json
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python ./unilm/decode_seq2seq.py \
  --fp16 --model_type unilm --tokenizer_name unilm1.2-base-uncased --input_file ./data/data.json --split validation --do_lower_case \
  --model_path ./unilm/checkpoint/ckpt-22500 --max_seq_length 512 --max_tgt_length 48 --batch_size 32 --beam_size 5 \
  --length_penalty 0 --forbid_duplicate_ngrams --mode s2s --forbid_ignore_word "."