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

## Training using UserLibri TTS data
1. Place libri_prefinetuned.pt in
```
/path/to/icefall/egs/userlibri/ASR/models
```
2. Unzip UserLibri data in /DB/
3. Prepare UserLibri data

```bash
cd /path/to/icefall/egs/userlibri/ASR
. ./prepare_tts_gen.sh
. ./prepare_userlibri.sh
```

4. Personalize model with userlibri TTS data & inference on test sets
```bash
cd /path/to/icefall/egs/userlibri/ASR
. ./run_individual_test_clean_tts.sh
. ./run_individual_test_other_tts.sh
```