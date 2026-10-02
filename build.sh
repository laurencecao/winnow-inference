
export CUDA_HOME=/usr/local/cuda-13  # 例如 /usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDACXX=$CUDA_HOME/bin/nvcc

https_proxy=10.0.2.15:7890 python3 scripts/build.py --jobs 3
