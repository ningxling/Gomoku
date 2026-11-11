#!/bin/bash
# 看门狗：进程挂了自动拉起；训练卡死（45 分钟没有新轮次）自动重启。
# 注：每轮要 15~20 分钟（sims 120 + 128 通道 + 对抗陪练），阈值不能太紧，
# 用法：setsid nohup ./watchdog.sh >> watchdog.log 2>&1 < /dev/null &
cd /home/rocky/gomoku || exit 1

log() { echo "[$(date '+%F %T')] $*" >> watchdog.log; }

log "看门狗启动（进程守护 + 卡死检测）"
prev_it=""
stall=0

while true; do
    # 1) 训练进程
    if ! pgrep -f '^python3 train.py' > /dev/null; then
        log "训练不在运行 -> 重启"
        ./restart_vm.sh >> watchdog.log 2>&1
        prev_it=""; stall=0
    else
        # 2) 卡死检测：连续 9 次（45 分钟）没有新轮次就重启
        it=$(grep -a 'iter' train.log 2>/dev/null | tail -n 1 | sed -E 's/iter +([0-9]+).*/\1/')
        if [ -n "$it" ] && [ "$it" = "$prev_it" ]; then
            stall=$((stall + 1))
            log "训练停在 iter $it（第 $stall 次检查）"
        else
            stall=0
            prev_it="$it"
        fi
        if [ "$stall" -ge 9 ]; then
            log "iter $it 已 45 分钟无进展 -> 判定卡死，重启"
            ./restart_vm.sh >> watchdog.log 2>&1
            prev_it=""; stall=0
        fi
    fi

    # 3) 存档循环
    if ! pgrep -f '^python3 snapshot_ckpt.py' > /dev/null; then
        log "存档循环不在运行 -> 拉起"
        setsid nohup python3 snapshot_ckpt.py --interval 1800 >> snapshot.log 2>&1 < /dev/null &
    fi

    # 4) 健康巡检
    if ! pgrep -f '^python3 monitor_health.py' > /dev/null; then
        log "健康巡检不在运行 -> 拉起"
        setsid nohup python3 monitor_health.py --interval 3600 >> monitor.log 2>&1 < /dev/null &
    fi

    # 5) 代际闸门（每 50 轮让最新权重挑战最佳权重）
    if ! pgrep -f '^python3 gate_watch.py' > /dev/null; then
        log "闸门调度不在运行 -> 拉起"
        setsid nohup python3 gate_watch.py --interval 30 --games 20 --sims 120 >> gate_watch.log 2>&1 < /dev/null &
    fi

    sleep 300
done
