## Hand-by-hand guidance for Setup MinkowskiEngine 0.4.3 Env

Note ME 0.5.4 is much faster than ME 0.4.3, recommend to use ME 0.5.4

## :fire: Code is coming soon!

### About us
We provide our environment info as follows:
- Hardware: Nvidia GeForce RTX TITAN XP (12 GB) * 8
- CUDA: 10.2
- PyTorch: 1.6
- Python: 3.7
- GCC/G++: 7.5.0


### Install basic python and pytorch environment
```bash
conda create -n cpcm_me043 python=3.7 -y
conda activate cpcm_me043
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch -y
```


### Setup CUDA env variable
May replace to your cuda path
```bash
export CUDA_HOME=/usr/local/cuda-10.2/
export PATH=${CUDA_HOME}/bin${PATH:+:${PATH}}
export CMAKE_CUDA_COMPILER=${CUDA_HOME}/bin/nvcc
```

### Install other libraries, which is needed for MinkowskiEngine 0.4.3
```bash
conda install mkl mkl-include tbb -c intel -y
```

### Install MinkowskiEngine 0.4.3
Get minkowski engine from official website and unzip it
```bash
mkdir env
cd env
wget https://github.com/NVIDIA/MinkowskiEngine/archive/refs/tags/v0.4.3.zip
unzip v0.4.3.zip
cd MinkowskiEngine-0.4.3
```

Install the MinkowskiEngine 0.4.3
```bash
python setup.py install
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