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

model_dir=pruned_transducer_stateless_conformer
ft_model=./$model_dir/C_0/epoch-30.pt
# datset: librispeech, ljspeech, userlibri
train_dataset="l2arctic"
test_dataset="l2arctic"
EMA=1
flel=9
fz_enc=False
fz_dec=False
fz_decemb=True
ctc_scale=0.0
lwf=False
l2=False
max_epoch=20
bookid_list=$(cat /DB/l2arctic/list.txt)
for bookid in $bookid_list; do
  individual=$bookid

  expdir=./$model_dir/C_l2arctictts_${individual}_fz
  if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    log "Stage 0: Train model"
    ./pruned_transducer_stateless_conformer/train.py \
        --wandb False \
        --train-dataset $train_dataset \
        --lwf $lwf \
        --train-individual $individual \
        --individual-bookid $bookid \
        --load-prefinetuned-model $ft_model \
        --use-pseudo-labels False \
        --on-the-fly-pseudo-labels False \
        --start-epoch 1 \
        --world-size 4 \
        --num-epochs $max_epoch \
        --exp-dir $expdir \
        --num-buckets 2 \
        --max-duration 400 \
        --use-fp16 1 \
        --additional-block True \
        --prune-range 10 \
        --context-size 2 \
        --ctc-loss-scale 0.0 \
        --ema-alpha ${EMA} \
        --layer-average-start-idx -1 \
        --freeze-lower-encoder-layers $flel \
        --freeze-decoder-embedding-layers False \
        --freeze-encoder $fz_enc \
        --freeze-decoder $fz_dec \
        --freeze-joiner False \
        --enable-musan True \
        --enable-spec-aug True
    
    mv $expdir/epoch-$max_epoch.pt $expdir/last-epoch.pt
    rm -rf $expdir/epoch-*
  fi

  if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Decoding"
    # modified_beam_search, greedy_search, ctc_greedy_search
    for model_name in "best-valid-wer.pt"; do
        for method in modified_beam_search; do
            ./pruned_transducer_stateless_conformer/decode.py \
            --test-dataset $test_dataset \
            --decode-individual $individual \
            --gen-pseudo-label False \
            --model-name $model_name \
            --exp-dir $expdir \
            --num-buckets 2 \
            --max-duration 400 \
            --decoding-method $method \
            --max-sym-per-frame 1
        done
        mv $expdir/$method/wer-summary-$individual-beam_size_4-epoch-30-avg-9-$method-beam-size-4-use-averaged-model.txt $expdir/$method/wer-$model_name-summary-$individual-beam_size_4-epoch-30-avg-9-$method-beam-size-4-use-averaged-model.txt
    done

    for model_name in "epoch-30.pt"; do
        expdir=pruned_transducer_stateless_conformer/C_0
        for method in modified_beam_search; do
            ./pruned_transducer_stateless_conformer/decode.py \
            --test-dataset $test_dataset \
            --decode-individual $individual \
            --gen-pseudo-label False \
            --model-name $model_name \
            --exp-dir $expdir \
            --num-buckets 2 \
            --max-duration 400 \
            --decoding-method $method \
            --max-sym-per-frame 1
        done
    done
fi
done