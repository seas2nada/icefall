# Setup virtual environment
. ./setup_cuda_env.sh /usr/local/cuda
. ./setup_anaconda.sh venv base 3.8.13
. ./activate_python.sh && pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# install k2
rm -rf k2
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
