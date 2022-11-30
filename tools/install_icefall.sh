# Setup virtual environment
. ./setup_cuda_env.sh /usr/local/cuda
. ./setup_anaconda.sh venv base 3.8.13
. ./activate_python.sh && conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch -c conda-forge

# install k2
git clone https://github.com/k2-fsa/k2.git
cd k2
export K2_MAKE_ARGS="-j6"
python3 setup.py install

# install lhostse
cd ../
pip install git+https://github.com/lhotse-speech/lhotse

# install icefall
cd ../
pip install -r requirements.txt
icefall_dir=$PWD
echo 'export PYTHONPATH=$PYTHONPATH:'$icefall_dir >> ~/.bashrc
echo 'export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python' >> ~/.bashrc
source ~/.bashrc
conda install -c conda-forge gcc=12.1.0
