#!/bin/bash
cd /home/rocky/gomoku
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
# 先在 9x9 上把棋力拉起来（网页默认就是 9x9，样本效率最高）
# sims 60 + buffer 8000 + epochs 3：每轮约 15 秒，每小时约 30 万样本
# 想回到 19x19 课程式训练，把 --curriculum 换成 "9:0,13:30,15:100,19:250"
exec python3 train.py --curriculum "9:0" --sims 60 --episodes 12 --buffer 8000 \
                      --epochs 3 --teacher-games 3 --teacher-time 0.10
