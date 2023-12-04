## HYnet ver. Icefall Installation

### Install docker (Optional)
```bash
cd /path/to/icefall/docker/HYnet_docker
make _build
make run
```

### Login to icefall docker (Optional)
```bash
ssh icefall@localhost -p 32778
PW: if2022
```

### Install icefall
```bash
cd /path/to/icefall/tools
. ./install_icefall.sh
```

### Check icefall installation
```bash
cd /path/to/icefall/tools
. ./check_install.sh
```

### Install toolkits
```bash
cd /path/to/icefall/tools
. ./install_espnet.sh

cd /path/to/icefall/tools
. ./install_fairseq.sh
```

## Prepare data
1. Place libri_prefinetuned.pt in
```
/path/to/icefall/egs/userlibri/ASR/models
```
2. Place lang_bpe_500 in data
3. Unzip UserLibri data in /DB/
4. Prepare UserLibri data recipe

```bash
cd /path/to/icefall/egs/userlibri/ASR
. ./prepare.sh
```

## Training using UserLibri TTS data [Optional]
Personalize model with userlibri TTS data & inference on test sets
```bash
cd /path/to/icefall/egs/userlibri/ASR
. ./prepare_tts_gen.sh
bash ./run_individual_test_clean_tts.sh
bash ./run_individual_test_other_tts.sh
```

## Infer with pre-trained models
1. Place pre-trained models (Ex. if using other_models)
```bash
mv other_models /path/to/pruned_transducer_stateless_d2v/pretrained_models
```
2. Infer
```bash
cd /path/to/icefall/egs/userlibri/ASR
sudo chmod -R +777 ./*
bash infer.sh
bash avg_wers.sh
```
3. Check decoding_result.txt