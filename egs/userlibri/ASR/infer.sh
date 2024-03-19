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

model_dir=pruned_transducer_stateless_d2v
ft_model=$PWD/models/libri_prefinetuned.pt
# datset: librispeech, ljspeech, userlibri
train_dataset="userlibri"
test_dataset="userlibri"
model_list=$(cat $model_dir/pretrained_models/list.txt)

for model in $model_list; do
    individual=$(echo $model | awk -F '_' '{print $2}')
    rm -rf $model_dir/pretrained_models/$model/modified_beam_search*
    expdir=$model_dir/pretrained_models/$model

    for model_name in "last-epoch.pt"; do
        for method in modified_beam_search; do
            ./${model_dir}/decode.py \
            --test-dataset $test_dataset \
            --decode-individual $individual \
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
        mv $expdir/$method/wer-summary-$individual-beam_size_4-epoch-30-avg-9-$method-beam-size-4-use-averaged-model.txt $expdir/$method/wer-$model_name-summary-$individual-beam_size_4-epoch-30-avg-9-$method-beam-size-4-use-averaged-model.txt
    done
done