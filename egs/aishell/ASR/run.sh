. ../../../tools/activate_python.sh
export PYTHONPATH=/home/ubuntu/Workspace/icefall/:$PYTHONPATH
torchrun --nproc_per_node 4 ./whisper_repo/train.py \
  --max-duration 200 \
  --exp-dir whisper_exp_large_v3 \
  --model-name large-v3 \
  --manifest-dir data/fbank_whisper \
  --deepspeed \
  --deepspeed_config ./whisper_repo/ds_config_zero1.json
