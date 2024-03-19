#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
. ../../../tools/activate_python.sh

set -eou pipefail

dir="pruned_transducer_stateless_d2v_v2/M_0/modified_beam_search"
out_dir="userlibri_results"

if [ ! -d "$out_dir" ]; then
  mkdir -p $out_dir
fi

. shared/parse_options.sh || exit 1

ls $dir | grep wer-summary > temp_ls.txt
rm -rf $out_dir/spk_results.txt
rm -rf $out_dir/book_results.txt

sum_spk=0
sum_book=0
while read line; do
    WER=`cat $dir/$line | tail -1 | awk -F " " '{print $2}'`
    fname=`echo $line | awk -F "-" '{print $3"-"$4}'`
    split=`echo $line | awk -F "-" '{print $3}'`
    
    if [ $split == "book" ]; then
        echo "$WER $fname" >> $out_dir/book_results_.txt
        sum_book=$(echo "$sum_book + $WER" | bc)
    elif [ $split == "speaker" ]; then
        echo "$WER $fname" >> $out_dir/spk_results_.txt
        sum_spk=$(echo "$sum_spk + $WER" | bc)
    fi
done < temp_ls.txt

cat $out_dir/book_results_.txt | sort -n > $out_dir/book_results.txt
cat $out_dir/spk_results_.txt | sort -n > $out_dir/spk_results.txt

spk_linnum=`wc -l $out_dir/spk_results.txt | awk -F " " '{print $1}'`
book_linnum=`wc -l $out_dir/book_results.txt | awk -F " " '{print $1}'`
mean_spk=$(echo "scale=2; $sum_spk / $spk_linnum" | bc)
mean_book=$(echo "scale=2; $sum_book / $spk_linnum" | bc)

echo "$mean_book MEAN" >> $out_dir/book_results.txt
echo "$mean_spk MEAN" >> $out_dir/spk_results.txt

rm -rf $out_dir/book_results_.txt
rm -rf $out_dir/spk_results_.txt
rm -rf temp_ls.txt