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

expdir=./pruned_transducer_stateless4_EP/libri_960
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Train model"
  ./pruned_transducer_stateless4_EP/train.py \
    --world-size 1 \
    --num-epochs 30 \
    --start-epoch 2 \
    --use-fp16 1 \
    --exp-dir $expdir \
    --full-libri 1 \
    --max-duration 550
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Decoding"
  for method in modified_beam_search; do
    ./pruned_transducer_stateless4_EP/decode.py \
      --use-averaged-model True \
      --epoch 11 \
      --avg 2 \
      --exp-dir $expdir \
      --max-duration 600 \
      --decoding-method modified_beam_search \
      --beam-size 4
  done
fi