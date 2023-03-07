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
individual="speaker-1995"
bookid="15265"
test_dataset="userlibri"

flel=9
fz_dec=False
fz_decemb=True
max_epoch=20
expdir=./$model_dir/M_${individual}_book-${bookid}_EMA499_fz-lowenc$flel-dec$fz_dec-decemb$fz_decemb
pn=UserLibri_iter0
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Train model"
  ./pruned_transducer_stateless_d2v_dhver/train.py \
          --wandb False \
          --train-individual $individual \
          --individual-bookid $bookid \
          --use-pseudo-labels False \
          --on-the-fly-pseudo-labels False \
          --pseudo-name $pn \
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
          --ctc-loss-scale 0.2 \
          --peak-dec-lr 0.04175 \
          --peak-enc-lr 0.0003859 \
          --update-ema True \
          --layer-average-start-idx -1 \
          --freeze-lower-encoder-layers $flel \
          --freeze-decoder-embedding-layers False \
          --freeze-encoder False \
          --freeze-decoder $fz_dec \
          --freeze-joiner False \
          --enable-musan False
  
  mv $expdir/epoch-$max_epoch.pt $expdir/last-epoch.pt
  rm -rf $expdir/epoch-*
fi
# --peak-dec-lr 0.04175 \
# --peak-enc-lr 0.0003859 \
# TODO:
# 1. Cross validation
# 2. Low rank adaptation
# 3. Importance sampling of parameters

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Decoding"
  # modified_beam_search, greedy_search, ctc_greedy_search
  for model_name in "best-valid-wer.pt" "last-epoch.pt"; do
    # expdir=./$model_dir/M_${individual}_book-${bookid}_EMA499_fz-lowenc9-dec
    expdir=./$model_dir/M_0
    model_name="libri_prefinetuned.pt"
    test_dataset="librispeech"
    ./pruned_transducer_stateless_d2v_dhver/decode_rnnlm.py \
    --test-dataset $test_dataset \
    --decode-individual $individual \
    --gen-pseudo-label False \
    --input-strategy AudioSamples \
    --enable-spec-aug False \
    --additional-block True \
    --model-name $model_name \
    --exp-dir $expdir \
    --encoder-type d2v \
    --encoder-dim 768 \
    --decoder-dim 768 \
    --joiner-dim 768 \
    --max-duration 600 \
    --decoding-method modified_beam_search_rnnlm_shallow_fusion \
    --beam 4 \
    --rnn-lm-scale 0.3 \
    --rnn-lm-exp-dir rnnlm_model \
    --rnn-lm-epoch 99 \
    --rnn-lm-avg 1 \
    --rnn-lm-num-layers 3 \
    --rnn-lm-tie-weights 1

    # for method in modified_beam_search; do
    #     ./pruned_transducer_stateless_d2v_dhver/decode.py \
    #     --test-dataset $test_dataset \
    #     --decode-individual $individual \
    #     --gen-pseudo-label False \
    #     --input-strategy AudioSamples \
    #     --enable-spec-aug False \
    #     --additional-block True \
    #     --model-name $model_name \
    #     --exp-dir $expdir \
    #     --num-buckets 2 \
    #     --max-duration 400 \
    #     --decoding-method $method \
    #     --max-sym-per-frame 1 \
    #     --encoder-type d2v \
    #     --encoder-dim 768 \
    #     --decoder-dim 768 \
    #     --joiner-dim 768
    # done
  done
fi