# usage:
# python3 utils/decoded_result_to_text_files.py input_file output_directory

import os
import sys
import re

input_file = sys.argv[1]
output_directory = sys.argv[2]
output_parent = os.path.join(output_directory, 'audio_data/speaker-wise-test')

if not os.path.exists(output_directory):
    os.makedirs(output_directory, exist_ok=True)

def find_file(root_dir, file_name):
    for root, dirs, files in os.walk(root_dir):
        if file_name in files:
            return os.path.join(root, file_name)
    return None

out_dict = {}
with open(input_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if "_sp" in line:
            continue
        # use regular expression to extract the filename and hyp list
        # EX) 2033-164914-0010-10:	hyp=['BY', 'ALLAH', 'AN', 'THOU', 'FETCH', 'HIM']
        match = re.search(r'(.*):\s*hyp=\[(.*)\]', line)
        if match:
            tname = '-'.join(line.split(':')[0].split('-')[:-1])
            utt_name = '-'.join(tname.split('-')[:-1])
            spk_name = utt_name.split('-')[0]
            dir_name = 'speaker-' + spk_name
            spk_dir = os.path.join(output_parent, dir_name)

            if utt_name not in out_dict.keys():
                out_dict[utt_name] = {}
                trn4utt = utt_name + ".trans.txt"
                output_file = find_file(spk_dir, trn4utt)
                out_dict[utt_name]["output_file"] = output_file
                out_dict[utt_name]["texts"] = {}
                if output_file is None:
                    raise NotImplementedError(f"{trn4utt} not in {spk_dir}")

            filename = match.group(1)
            hyp = match.group(2)
            hyp = hyp.replace("'", "").replace("[", "").replace("]", "").split(',')
            hyp_sentence = ' '.join(hyp).replace("  "," ")

            out_dict[utt_name]["texts"][tname] = hyp_sentence

for utt_name in out_dict.keys():
    output_file = out_dict[utt_name]["output_file"]
    texts_dict = out_dict[utt_name]["texts"]

    output_file_org = output_file + "_org"
    if not os.path.exists(output_file_org):
        os.rename(output_file, output_file_org)

    with open(output_file, 'w') as f:
        for tname in texts_dict.keys():
            sentence = tname + " " + texts_dict[tname] + "\n"
            f.write(sentence)