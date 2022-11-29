# Setup virtual environment
. ./setup_cuda_env.sh /usr/local/cuda
. ./setup_anaconda.sh venv base 3.8.13
. ./activate_python.sh && conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch -c conda-forge

