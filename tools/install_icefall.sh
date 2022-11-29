# Setup virtual environment
. ./setup_cuda_env.sh /usr/local/cuda
. ./setup_anaconda.sh venv base 3.8.13
. ./activate_python.sh && conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch -c conda-forge

# install k2
conda install -c k2-fsa -c pytorch -c conda-forge k2 python=3.8 cudatoolkit=11.3 pytorch=1.12.0

# install lhostse
pip install git+https://github.com/lhotse-speech/lhotse

# install icefall
cd ../
pip install -r requirements.txt
export PYTHONPATH="$PYTHONPATH:$PWD"
conda install -c conda-forge gcc=12.1.0
