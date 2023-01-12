# Dependencies
sudo apt install libfftw3-dev libopenmpi-dev
sudo apt install build-essential cmake
sudo apt install libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev
sudo apt install zlib1g-dev libbz2-dev liblzma-dev
sudo apt install pkg-config

# activate venv
. ./activate_python.sh

# build fairseq
rm -rf fairseq
git clone https://github.com/facebookresearch/fairseq.git; cd fairseq
sed -i 's/torch>=1.13/torch>=1.10/g' setup.py
pip install --editable ./
cd ../

# Install arrayfire
wget https://arrayfire.s3.amazonaws.com/3.8.0/ArrayFire-v3.8.0_Linux_x86_64.sh
sudo chmod -R +777 ./ArrayFire-v3.8.0_Linux_x86_64.sh
./ArrayFire-v3.8.0_Linux_x86_64.sh

# Install flashlight
# Piz install mkl before excute above lines
rm -rf flashlight
export PATH=/usr/local/cuda-11.3/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
git clone https://github.com/flashlight/flashlight.git && cd flashlight
git reset --hard 03c51129f320eed7ff0d416f7e8291a029439039
mkdir -p build && cd build

## install kenlm
git clone https://github.com/kpu/kenlm.git
cd kenlm
mkdir -p build && cd build
cmake .. 
make -j 16

# error "Could NOT find NCCL": https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html
cmake .. -DCMAKE_BUILD_TYPE=Release -DFL_BACKEND=CUDA \
-DArrayFire_DIR=$PWD/../../arrayfire/share/ArrayFire/cmake/
make -j$(nproc)
sudo make install

## flashlight binding
cd ../../../../flashlight/bindings/python
python3 setup.py install

# install tensorboardX
pip install tensorboardX
