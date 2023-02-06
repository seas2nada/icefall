#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
. ../../../tools/activate_python.sh

set -eou pipefail

stage=0
stop_stage=100
world_size=4

. shared/parse_options.sh || exit 1

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

ft_model=./pruned_transducer_stateless_d2v_dhver/M_0/libri_prefinetuned.pt

model_dir=d2v_transducer_text_encoder_finetuning
individual="speaker-1995"

expdir=./$model_dir/M_${individual}_text_ft_ce
pn=UserLibri_iter1
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
log "Stage 0: Train model"
./d2v_transducer_text_encoder_finetuning/train.py \
        --wandb False \
        --text-path "data/texts/15265_lm_train.txt" \
        --load-prefinetuned-model $ft_model \
        --input-strategy AudioSamples \
        --enable-spec-aug False \
        --multi-optim True \
        --start-epoch 1 \
        --world-size 4 \
        --num-epochs 100 \
        --exp-dir $expdir \
        --num-buckets 2 \
        --max-duration 100 \
        --freeze-finetune-updates 0 \
        --encoder-dim 768 \
        --decoder-dim 768 \
        --joiner-dim 768 \
        --use-fp16 1 \
        --accum-grads 4 \
        --encoder-type d2v \
        --additional-block True \
        --prune-range 10 \
        --context-size 2 \
        --ctc-loss-scale 0.2 \
        --peak-dec-lr 0.04175 \
        --peak-enc-lr 0.0003859 \
        --update-ema False \
        --layer-average False
fi

rm -rf $expdir/epoch-*