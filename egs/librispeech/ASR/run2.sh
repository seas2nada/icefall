#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
. ../../../tools/activate_python.sh

set -eou pipefail

stage=0
stop_stage=100

model=pruned_transducer_stateless_w2v
world_size=4

. shared/parse_options.sh || exit 1

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Train model"
  ./pruned_transducer_stateless_w2v_v2/train.py \
    --wandb false \
    --input-strategy AudioSamples \
    --enable-spec-aug False \
    --multi-optim True \
    --world-size 4 \
    --num-epochs 30 \
    --start-epoch 1 \
    --full-libri 1 \
    --exp-dir ./pruned_transducer_stateless_w2v_v2/exp_multioptim \
    --max-duration 150 \
    --freeze-finetune-updates 2000 \
    --use-fp16 0 \
    --peak-enc-lr 0.001 \
    --peak-dec-lr 0.05 \
    --accum-grads 1 \
    --encoder-type w2v \
    --additional-block True \
    --encoder-dim 768 \
    --decoder-dim 768 \
    --joiner-dim 768 \
    --prune-range 20 \
    --context-size 2 \
    --ctc-loss-scale 0.2 \
    --w2v-url "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt" \
    --freeze-param "encoder.encoders.mask_emb" "encoder.encoders.feature_extractor" "encoder.encoders.quantizer" "encoder.encoders.project_q"
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Decoding"
  decoding_method="greedy_search"  # "fast_beam_search", "modified_beam_search"

  for chunk in 16; do
    for left in 64; do
      ./pruned_transducer_stateless_w2v_v2/decode.py \
        --input-strategy AudioSamples \
        --epoch 30 \
        --avg 5 \
        --exp-dir ./pruned_transducer_stateless_w2v_v2/exp_multioptim \
        --max-duration 600 \
        --decoding-method modified_beam_search \
        --beam-size 4 \
        --encoder-type w2v \
        --w2v-url "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt" \
        --encoder-dim 768 \
        --decoder-dim 768 \
        --joiner-dim 768 \
        --prune-range 20 \
        --context-size 2 \ 
        --ctc-loss-scale 0.2 \
        --additional-block True
    done
  done
fi