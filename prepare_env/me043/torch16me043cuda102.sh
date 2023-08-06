conda create -n cpcm_me043 python=3.7 -y
conda activate cpcm_me043

conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch -y

export CUDA_HOME=/usr/local/cuda-10.2
export PATH=${CUDA_HOME}/bin${PATH:+:${PATH}}
export CMAKE_CUDA_COMPILER=${CUDA_HOME}/bin/nvcc

# pip install mkl-include -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install tbb-2021.7.1-py2.py3-none-manylinux1_x86_64.whl
# pip install mkl-2021.4.0-py2.py3-none-manylinux1_x86_64.whl
conda install mkl mkl-include tbb -c intel -y

wget https://github.com/NVIDIA/MinkowskiEngine/archive/refs/tags/v0.4.3.zip
unzip v0.4.3.zip
cd MinkowskiEngine-0.4.3
python setup.py install

pip install -r requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple
