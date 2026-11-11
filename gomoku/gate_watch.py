"""闸门调度：盯着 train.log 的轮数，每 N 轮跑一次 gate.py。

用法：
    python3 gate_watch.py --interval 50            # 每 50 轮评估一次
    python3 gate_watch.py --interval 50 --once     # 只评估一次（手动触发）

设计为长期后台运行：训练照常写 latest.pt，闸门只负责"选最佳/报警"，
不会中断也不会冻结训练。
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "checkpoints", "gate_watch.json")
RE_ITER = re.compile(r"^iter\s+(\d+) \|")


def ckpt_iter(path):
    """latest.pt 里记录的轮数（训练重启时日志会比权重"超前"）。"""
    try:
        import torch
        return int(torch.load(path, map_location="cpu").get("iter", 0))
    except Exception:
        return 0


def last_iter():
    """train.log 里最后一条完成的轮次。"""
    path = os.path.join(HERE, "train.log")
    if not os.path.exists(path):
        return 0
    last = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = RE_ITER.match(line)
            if m:
                last = int(m.group(1))
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=50)
    ap.add_argument("--poll", type=int, default=300)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--sims", type=int, default=120)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()

    state = {"next": 0}
    if os.path.exists(STATE):
        try:
            state = json.load(open(STATE, encoding="utf-8"))
        except Exception:
            pass

    while True:
        cur = last_iter()
        if state["next"] == 0:
            base = max(ckpt_iter(os.path.join(HERE, "checkpoints", "latest.pt")), 0)
            state["next"] = min(cur, base) + a.interval if base else cur + a.interval
            json.dump(state, open(STATE, "w", encoding="utf-8"))
            print(f"[watch] 当前 {cur} 轮，下一次评估在 {state['next']} 轮", flush=True)
        if cur >= state["next"]:
            print(f"[watch] 到达 {cur} 轮，触发闸门评估", flush=True)
            subprocess.run([sys.executable, os.path.join(HERE, "gate.py"),
                            "--games", str(a.games), "--sims", str(a.sims)],
                           cwd=HERE)
            state["next"] = cur + a.interval
            json.dump(state, open(STATE, "w", encoding="utf-8"))
        if a.once:
            return
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
