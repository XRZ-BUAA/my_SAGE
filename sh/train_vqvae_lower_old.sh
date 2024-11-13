#!/bin/bash
module load cuda/11.8 cudnn/8.7.0_cuda11.x
module list
# 加载 conda
source /data/run01/scw6dwh/miniconda3/etc/profile.d/conda.sh  # 确保加载了 conda（修改为你的 conda 路径）

# 激活 conda 环境
conda activate sage2

# 进入工作目录
cd /data/run01/scw6dwh/sxh/SAGE

# 运行 Python 脚本
python train_vqvae.py --cfg config_vqvae/lower_vqvae_our.yaml