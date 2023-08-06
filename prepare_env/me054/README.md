## Hand-by-hand guidance for Setup MinkowskiEngine 0.5.4 Env

Note ME 0.5.4 is much faster than ME 0.4.3, recommend to use ME 0.5.4

### About us
We provide our environment info as follows:
- Hardware: Nvidia GeForce RTX 3090 (24 GB) * 8
- CUDA: 11.6
- PyTorch: 1.10.1
- Python: 3.7
- GCC/G++: 9.4.0

### Training cost
- ScanNetV2: Nvidia GeForce RTX 3090 (24 GB) * 2, < 2 days
- S3DIS: Nvidia GeForce RTX 3090 (24 GB) * 1, < 1 days

### Install basic python and pytorch environment
```bash
conda create -n cpcm python=3.7 -y
conda activate cpcm
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y
```

### Install gcc/g++ 9 (Skip if already satisfied)
Check the gcc/g++ version, if gcc/g++ version is 9.x.x, skip the following steps
```bash
gcc -v 
g++ -v
```
Install gcc/g++ 9
```bash
sudo apt install gcc-9 g++-9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9
```
Check gcc/g++ 9 installed or not
```bash
gcc -v 
g++ -v
```

### Setup CUDA env variable
May replace to your cuda path
```bash
export CUDA_HOME=/usr/local/cuda-11.6/
export PATH=${CUDA_HOME}/bin${PATH:+:${PATH}}
export CMAKE_CUDA_COMPILER=${CUDA_HOME}/bin/nvcc
```

### Install other libraries
Minkowski Engine requires openblas and ninja to compile
```bash
sudo apt-get install libopenblas-dev
pip install ninja -i https://pypi.tuna.tsinghua.edu.cn/simple # for minkowski engine compile, ninja will speedup the compile
```

### Install MinkowskiEngine 0.5.4
Get minkowski engine from official website and unzip it
```bash
mkdir env
cd env
wget https://github.com/NVIDIA/MinkowskiEngine/archive/refs/tags/v0.5.4.zip
unzip v0.5.4.zip
cd MinkowskiEngine-0.5.4
```
Modify pybind/extern.hpp according to [the issue of me054](https://github.com/NVIDIA/MinkowskiEngine/issues/414)
```bash
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
```

Install the MinkowskiEngine 0.5.4
```bash
python setup.py install --blas=openblas --force_cuda
```

Go back to the root directory of CPCM
```bash
cd ../..
```

### Install requirements.txt
Finally, install the requirements.txt
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```