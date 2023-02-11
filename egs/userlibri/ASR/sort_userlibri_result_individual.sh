#!/usr/bin/env bash

# Usage
# dir에 /path/to/model_dir
# out_dir에 /path/to/userlibri_results
# . ./sort_userlibri_result_individual.sh --model-dir $model_dir --decode-method $decode_method --out-dir $out_dir

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
. ../../../tools/activate_python.sh

model_dir="pruned_transducer_stateless_d2v_dhver/"
decode_method="modified_beam_search"
out_dir="userlibri_results/M_spkwise_PL_LA"
prefix="M"
suffix="pn"

set -eou pipefail

. shared/parse_options.sh || exit 1

if [ ! -d "$out_dir" ]; then
  mkdir -p $out_dir
fi

rm -rf $out_dir/spk_results.txt

spk_list=$(cat speaker_list.txt)

sum_spk=0
for spk in $spk_list; do
    lline=$(cat $model_dir/${prefix}_${spk}_${suffix}/$decode_method/wer-summary-${spk}-beam_size_4-epoch-30-avg-9-modified_beam_search-beam-size-4-use-averaged-model.txt | tail -1)
    WER=$(echo $lline | awk -F " " '{print $2}')
    echo $WER $spk >> $out_dir/spk_results_.txt
    sum_spk=$(echo "$sum_spk + $WER" | bc)
done

sort -n $out_dir/spk_results_.txt > $out_dir/spk_results.txt
rm -rf $out_dir/spk_results_.txt

spk_linnum=`wc -l $out_dir/spk_results.txt | awk -F " " '{print $1}'`
mean_spk=$(echo "scale=2; $sum_spk / $spk_linnum" | bc)

echo "$mean_spk MEAN" >> $out_dir/spk_results.txt