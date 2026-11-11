#!/bin/bash
# 一键启动/续起：训练 + 权重存档 + 健康巡检（都会先检查是否已在跑，重复执行无害）
cd /home/rocky/gomoku || exit 1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

if pgrep -f '^python3 train.py' > /dev/null; then
    echo "训练已在运行: $(pgrep -af '^python3 train.py')"
else
    setsid nohup ./run_train.sh >> train.log 2>> train.err < /dev/null &
    sleep 2
    echo "训练已启动: $(pgrep -af '^python3 train.py')"
fi

if pgrep -f '^python3 snapshot_ckpt.py' > /dev/null; then
    echo "存档循环已在运行"
else
    setsid nohup python3 snapshot_ckpt.py --interval 1800 >> snapshot.log 2>&1 < /dev/null &
    sleep 1
    echo "存档循环已启动（每 30 分钟一份，最多 400 份）"
fi

if pgrep -f '^python3 monitor_health.py' > /dev/null; then
    echo "健康巡检已在运行"
else
    setsid nohup python3 monitor_health.py --interval 3600 >> monitor.log 2>&1 < /dev/null &
    sleep 1
    echo "健康巡检已启动（每小时写一行到 checkpoints/monitor.csv）"
fi
