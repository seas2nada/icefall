list=$(cat list.txt)
for l in $list; do
    cd $l/modified_beam_search/
    sname=$(echo $l | awk -F '_' '{print $2}')
    wer=$(cat wer-last-epoch.pt-summary-${sname}-beam_size_4-epoch-30-avg-9-modified_beam_search-beam-size-4-use-averaged-model.txt | tail -1 | awk -F '\t' '{print $2}')

    cd ../../

    echo $sname >> snames_temp.txt
    echo $wer >> wer_temp.txt
    echo $wer $sname >> wer_summary_temp.txt
done

curr_wer=0
for wer in $(cat wer_temp.txt); do
    curr_wer=$(python3 -c "print($curr_wer + $wer)")
done
num_spks=$(wc -l wer_temp.txt | awk -F ' ' '{print $1}')
avg_wer=$(python3 -c "print(round($curr_wer/$num_spks, 2))")

sort wer_summary_temp.txt > wer_summary_last.txt
echo "$avg_wer MEAN" >> wer_summary_last.txt

rm -rf wer_summary_temp.txt snames_temp.txt wer_temp.txt
