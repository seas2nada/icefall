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

model_dir=pruned_transducer_stateless_d2v_dhver
ft_model=./$model_dir/M_0/libri_prefinetuned.pt
# datset: librispeech, ljspeech, userlibri
train_dataset="ljspeech"
test_dataset="ljspeech"
EMA=1
flel=9
fz_enc=False
fz_dec=False
fz_decemb=True
ctc_scale=0.0
lwf=False
l2=False
max_epoch=20

expdir=$model_dir/M_0_to_LJttsdomain_fromLJvits_fz
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    log "Stage 0: Train model"
    ./pruned_transducer_stateless_d2v_dhver/train.py \
            --wandb False \
            --train-dataset $train_dataset \
            --lwf $lwf \
            --l2 $l2 \
            --use-pseudo-labels False \
            --on-the-fly-pseudo-labels False \
            --load-prefinetuned-model $ft_model \
            --input-strategy AudioSamples \
            --enable-spec-aug False \
            --multi-optim True \
            --start-epoch 1 \
            --world-size 4 \
            --num-epochs $max_epoch \
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
            --ctc-loss-scale $ctc_scale \
            --peak-dec-lr 0.04175 \
            --peak-enc-lr 0.0003859 \
            --ema-alpha ${EMA} \
            --layer-average-start-idx -1 \
            --freeze-lower-encoder-layers $flel \
            --freeze-decoder-embedding-layers False \
            --freeze-encoder $fz_enc \
            --freeze-decoder $fz_dec \
            --freeze-joiner False \
            --enable-musan True

    mv $expdir/epoch-$max_epoch.pt $expdir/last-epoch.pt
    rm -rf $expdir/epoch-*
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Decoding"
    # modified_beam_search, greedy_search, ctc_greedy_search
    for model_name in "best-valid-wer.pt"; do
        for method in modified_beam_search; do
            ./pruned_transducer_stateless_d2v_dhver/decode_LJSpeech.py \
            --gen-pseudo-label False \
            --input-strategy AudioSamples \
            --enable-spec-aug False \
            --additional-block True \
            --model-name $model_name \
            --exp-dir $expdir \
            --num-buckets 2 \
            --max-duration 400 \
            --decoding-method $method \
            --max-sym-per-frame 1 \
            --encoder-type d2v \
            --encoder-dim 768 \
            --decoder-dim 768 \
            --joiner-dim 768
        done
    done
fi