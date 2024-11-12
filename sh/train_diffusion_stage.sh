#!/bin/bash
# 加载 conda
source /data/run01/scw6dwh/miniconda3/etc/profile.d/conda.sh  # 确保加载了 conda（修改为你的 conda 路径）

# 激活 conda 环境
conda activate hoi

# 进入工作目录
cd /data/run01/scw6dwh/sxh/SAGE

# 运行 Python 脚本
python train_first.py --cfg config_diffusion/first_ours.yaml

python train_second.py --cfg config_diffusion/second_ours.yaml