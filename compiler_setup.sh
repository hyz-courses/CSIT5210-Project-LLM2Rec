export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
# CUDA 仍用 NVHPC 提供的
export CUDA_HOME=/cm/shared/apps/nvhpc/23.11/Linux_x86_64/23.11/cuda/12.3
export PATH=${CUDA_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

# 验证
$CXX --version        # 应显示 gcc (GCC) 11.x
nvcc --version        # 仍应是 12.3
