#!/bin/bash
cd /home/rocky/gomoku
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
# 纯 19x19，样本全部落在目标棋盘
# episodes 8：每轮产量匹配 buffer；max-moves 170：砍掉“填格子”式的超长对局
# teacher-games 6：掺入分胜负的示范，避免价值头只会学“全是和棋”
# temp-moves 24 + temp 0.25：前 24 手采样，但用低温度——温度 1.0 会让对局 10 手就崩
#   （实测：τ=1.0 平均 12 手、τ=0.5 平均 20 手、τ=0.25 平均 36 手，且胜负颜色均衡）
# sims 120：19x19 上 60 次模拟看不出“活三 -> 活四”这种两级威胁，加到 120 让搜索能看 4~5 层
# 想切回混训：--mix-sizes "9:1,19:2"；想只跑 9x9：run_train_9x9.sh
exec python3 train.py --curriculum "19:0" --channels 128 --sims 120 --episodes 8 --buffer 8000 \
                      --epochs 3 --teacher-games 6 --teacher-time 0.10 \
                      --tactical-guard 1 --max-moves 170 --temp-moves 24 --temp 0.25 --spar-games 6 --spar-ratio 0.25
