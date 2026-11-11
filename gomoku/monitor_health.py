"""VM 上的训练健康巡检：每小时把关键指标写进 checkpoints/monitor.csv。

不依赖本地电脑、也不依赖 DSH 会话——只要 VM 活着、这个循环活着，就有连续的历史。
以后回看"什么时候开始好转/恶化"直接看这个 CSV。

用法（在 /home/rocky/gomoku 下）:
    setsid nohup python3 monitor_health.py --interval 3600 >> monitor.log 2>&1 < /dev/null &
"""
import argparse
import csv
import os
import re
import subprocess
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "train.log")
OUT = os.path.join(HERE, "checkpoints", "monitor.csv")

RE_ITER = re.compile(r"iter\s+(\d+)\s*\|\s*(\S+)\s*\|.*?loss\s+([\d.]+)\s*"
                     r"\(policy\s+([\d.]+)\s+value\s+([\d.]+)\).*?\|\s*([\d.]+)s")
RE_GAME = re.compile(r"对局: 黑胜 (\d+) / 白胜 (\d+) / 和 (\d+) / 截断 (\d+) \| 平均 (\d+) 手")
RE_GUARD = re.compile(r"战术护栏命中 (\d+)/(\d+) 手")


def tail_log(nbytes=262144):
    try:
        with open(LOG, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - nbytes))
            return f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return []


def find(patterns, lines):
    """从后往前找，返回第一组匹配。"""
    for line in reversed(lines):
        for p in patterns:
            m = p.search(line)
            if m:
                return m
    return None


def running():
    try:
        out = subprocess.run(["pgrep", "-fc", "^python3 train.py"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip() or 0)
    except Exception:                                   # noqa: BLE001
        return -1


def loadavg():
    try:
        with open("/proc/loadavg") as f:
            return f.read().split()[0]
    except OSError:
        return ""


def sample():
    lines = tail_log()
    it = find([RE_ITER], lines)
    gm = find([RE_GAME], lines)
    gd = find([RE_GUARD], lines)
    row = [datetime.now().strftime("%Y-%m-%d %H:%M")]
    if it:
        row += [it.group(1), it.group(2), it.group(3), it.group(4), it.group(5), it.group(6)]
    else:
        row += ["", "", "", "", "", ""]
    if gm:
        row += [gm.group(1), gm.group(2), gm.group(3), gm.group(4), gm.group(5)]
    else:
        row += ["", "", "", "", ""]
    row += [f"{gd.group(1)}/{gd.group(2)}" if gd else "", running(), loadavg()]
    return row


def main():
    ap = argparse.ArgumentParser(description="训练健康巡检")
    ap.add_argument("--interval", type=int, default=3600, help="采样间隔秒数")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    header = ["时间", "轮数", "棋盘", "loss", "policy", "value", "单轮秒",
              "黑胜", "白胜", "和", "截断", "平均手数", "护栏", "训练进程", "负载"]
    new = not os.path.exists(OUT)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    if args.once:
        row = sample()
        with open(OUT, "a", newline="", encoding="utf-8-sig" if new else "utf-8") as f:
            wr = csv.writer(f)
            if new:
                wr.writerow(header)
            wr.writerow(row)
        print(",".join(str(x) for x in row))
        return 0

    print(f"开始巡检：每 {args.interval} 秒一次 -> {OUT}", flush=True)
    while True:
        try:
            row = sample()
            new = not os.path.exists(OUT)
            with open(OUT, "a", newline="", encoding="utf-8-sig" if new else "utf-8") as f:
                wr = csv.writer(f)
                if new:
                    wr.writerow(header)
                wr.writerow(row)
            print(",".join(str(x) for x in row), flush=True)
        except Exception as e:                          # noqa: BLE001
            print(f"巡检失败: {e}", flush=True)
        time.sleep(max(300, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
