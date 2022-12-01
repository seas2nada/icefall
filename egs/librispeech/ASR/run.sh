#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
. ../../../tools/activate_python.sh

set -eou pipefail

stage=0
stop_stage=0

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
  ./${model}/train.py \
    --input-strategy AudioSamples \
    --enable-spec-aug false \
    --world-size ${world_size} \
    --num-epochs 30 \
    --start-epoch 1 \
    --full-libri 1 \
    --exp-dir ./pruned_transducer_stateless_w2v/exp \
    --max-duration 150 \
    --use-fp16 0 \
    --encoder-type w2v \
    --encoder-dim 768 \
    --decoder-dim 512 \
    --joiner-dim 512  \
    --w2v-url "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt" \
    --freeze-param "encoder.encoders.mask_emb" "encoder.encoders.feature_extractor" "encoder.encoders.post_extract_proj" "encoder.encoders.quantizer" "encoder.encoders.project_q"
fi