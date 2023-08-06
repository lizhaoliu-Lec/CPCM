# Install basic python and pytorch environment
conda create -n cpcm python=3.7 -y
conda activate cpcm
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y


# Note gcc and g++ should be version of gcc-9, g++-9
sudo apt install gcc-9 g++-9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9

# Check gcc/g++ 9 installed or not
gcc -v
g++ -v

# Setup CUDA env variable
export CUDA_HOME=/usr/local/cuda-11.6/
export PATH=${CUDA_HOME}/bin${PATH:+:${PATH}}
export CMAKE_CUDA_COMPILER=${CUDA_HOME}/bin/nvcc

# openblas also required
sudo apt-get install libopenblas-dev

# compile minkowski engine
# assert nvcc -V show cuda version is 11.x
pip install ninja -i https://pypi.tuna.tsinghua.edu.cn/simple # for minkowski engine compile, ninja will speedup the compile

# get minkowski engine
mkdir env
cd env
wget https://github.com/NVIDIA/MinkowskiEngine/archive/refs/tags/v0.5.4.zip
unzip v0.5.4.zip
cd MinkowskiEngine-0.5.4

# modify pybind/extern.hpp as https://github.com/NVIDIA/MinkowskiEngine/issues/414
vim pybind/extern.hpp
# (1) set line number by :set nu
# (2) go to line 765
# (3) change ".def(py::self == py::self);" into following content
"""
.def("__eq__", [](const minkowski::CoordinateMapKey &self, const minkowski::CoordinateMapKey &other)
                     {
                       return self == other;
                     });
      //.def(py::self == py::self);
"""
# (4) :wq

python setup.py install --blas=openblas --force_cuda

# after install MinkowskiEngine, return back
cd ..

# finally install the requirements.txt
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# for code migration, please see here
https://github.com/NVIDIA/MinkowskiEngine/wiki/Migration-Guide-from-v0.4.x-to-0.5.x
