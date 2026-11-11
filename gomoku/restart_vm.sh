#!/bin/bash
# 重启 VM 上的训练：备份当前权重 -> 停旧进程 -> 按 run_train.sh 重新拉起
# 日志是“追加”不是清空，这样看门狗夜里重启也不会丢掉历史
set -u
cd /home/rocky/gomoku || exit 1

cp -f checkpoints/latest.pt "checkpoints/latest.pt.pre_restart_$(date +%H%M)" 2>/dev/null

pkill -f '^python3 train.py' 2>/dev/null
sleep 3
if pgrep -f '^python3 train.py' > /dev/null; then
    echo "旧进程仍在，强制结束"
    pkill -9 -f '^python3 train.py' 2>/dev/null
    sleep 1
fi

echo "=== 重启 $(date '+%F %T') ===" >> train.log
setsid nohup ./run_train.sh >> train.log 2>> train.err < /dev/null &
sleep 5

echo "--- 进程 ---"
pgrep -af '^python3 train.py' || echo "启动失败"
echo "--- 日志 ---"
grep -a . train.log | tail -n 4
echo "--- 存档循环 ---"
pgrep -af '^python3 snapshot_ckpt.py' || echo "存档未运行（用 ./start_all.sh 拉起）"
