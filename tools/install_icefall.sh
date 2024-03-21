# Setup virtual environment
. ./setup_cuda_env.sh /usr/local/cuda
. ./setup_anaconda.sh venv base 3.10.12
. ./activate_python.sh && pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html

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

pip install wandb
