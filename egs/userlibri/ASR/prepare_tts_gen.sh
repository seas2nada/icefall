#!/usr/bin/env bash
. ../../../tools/activate_python.sh

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

nj=15
stage=-1
stop_stage=100

# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
#
#  - $dl_dir/UserLibri
#      You can find BOOKS.TXT, test-clean, train-clean-360, etc, inside it.
#      You can download them from https://www.openslr.org/12
#
#  - $dl_dir/lm
#      This directory contains the following files downloaded from
#       http://www.openslr.org/resources/11
#
#        - 3-gram.pruned.1e-7.arpa.gz
#        - 3-gram.pruned.1e-7.arpa
#        - 4-gram.arpa.gz
#        - 4-gram.arpa
#        - UserLibri-vocab.txt
#        - UserLibri-lexicon.txt
#        - UserLibri-lm-norm.txt.gz
#
#  - $dl_dir/musan
#      This directory contains the following directories downloaded from
#       http://www.openslr.org/17/
#
#     - music
#     - noise
#     - speech
dl_dir=/DB/
gen_dir=userlibri_test_clean_tts

. shared/parse_options.sh || exit 1

# vocab size for sentence piece models.
# It will generate data/lang_bpe_xxx,
# data/lang_bpe_yyy if the array contains xxx, yyy
vocab_sizes=(
  5000
  2000
  1000
  500
)

# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

# We assume that you have downloaded the UserLibri dataset
# You need to prepare UserLibri like below
# $dl_dir/UserLibri
# |-- $gen_dir
# |   |-- $tid_lm_train.txt
# |   |-- $tid
# |       |-- $tid_$uid.wav

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare UserLibri manifest"
  # We assume that you have downloaded the UserLibri corpus
  # to $dl_dir/UserLibri
  mkdir -p data/manifests
  if [ ! -e data/manifests/.tts_gen.done ]; then
    python prepare_tts_gen.py $dl_dir/UserLibri $gen_dir
    touch data/manifests/.tts_gen.done
  fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Compute fbank for userlibri"
  mkdir -p data/fbank
  if [ ! -e data/fbank/.tts_gen.done ]; then
    ./local/compute_fbank_tts_gen.py --data-dir $dl_dir/UserLibri --gen-dir $gen_dir
    touch data/fbank/.tts_gen.done
  fi

  if [ ! -e data/fbank/.userlibri-validated.done ]; then
    log "Validating data/fbank for speaker-wise userlibri"
    parts=`ls $dl_dir/UserLibri/$gen_dir --ignore "*.txt"`
    for part in ${parts[@]}; do
      python3 ./local/validate_manifest.py \
        data/fbank/userlibri_cuts_${part}.jsonl.gz
    done
  fi
fi