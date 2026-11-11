"""基准赛：本地 latest.pt vs 传统引擎，记录胜率趋势。

配合 sync_vm.py --bench N 使用：每回传一次新权重，就自动打一轮，结果追加到
checkpoints/bench_log.csv，方便看“强化学习追传统引擎”的曲线。

单独用也行：
    python bench_vs_teacher.py --games 6 --size 19 --sims 60 --time 0.1
"""
import argparse
import csv
import os
import sys
from datetime import datetime

import torch

from model import load_model
from teacher import NetAgent, TeacherAgent, match, score_to_elo

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description="强化学习权重 vs 传统引擎基准赛")
    ap.add_argument("--ckpt", default=os.path.join(HERE, "checkpoints", "latest.pt"))
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--size", type=int, default=19)
    ap.add_argument("--sims", type=int, default=60, help="网络每步 MCTS 模拟次数")
    ap.add_argument("--time", type=float, default=0.1, help="传统引擎每步秒数")
    ap.add_argument("--log", default=os.path.join(HERE, "checkpoints", "bench_log.csv"))
    ap.add_argument("--open-moves", type=int, default=4,
                    help="每局先随机下几手打散开局（0=完全确定性，多局会重放同一盘）")
    ap.add_argument("--tactical", action="store_true",
                    help="开启战术兜底：强制着法（成五/挡四/VCF）交给传统引擎直接走")
    ap.add_argument("--no-guard-three", action="store_true",
                    help="关闭活三护栏（对手有活三时不再限制候选点）")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    it = ckpt.get("iter")
    net = load_model(args.ckpt)
    net.eval()
    student = NetAgent(net, sims=args.sims, temperature=0.0,
                       tactical=args.tactical, tactical_time=args.time,
                       guard_three=not args.no_guard_three)
    teacher = TeacherAgent(time_limit=args.time, forbidden=True)

    w, l, d = match(teacher.policy, student.policy, args.games, args.size,
                    forbidden=True, tag_a="老师", tag_b="网络",
                    verbose=not args.quiet, opening_moves=args.open_moves)
    total = max(1, w + l + d)
    score = (l + 0.5 * d) / total          # 网络视角胜率
    row = [datetime.now().strftime("%Y-%m-%d %H:%M"), it, args.size, args.sims,
           args.games, w, l, d, f"{score:.3f}", f"{score_to_elo(1 - score):+.0f}",
           args.open_moves]
    new = not os.path.exists(args.log)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    # 新建时带 BOM，Excel 直接双击不乱码；之后追加用普通 utf-8
    with open(args.log, "a", newline="", encoding="utf-8-sig" if new else "utf-8") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(["时间", "轮数", "棋盘", "模拟次数", "局数",
                         "老师胜", "网络胜", "和", "网络胜率", "Elo差", "随机开局手数"])
        wr.writerow(row)
    print(f"基准赛: 第 {it} 轮, {args.size}x{args.size}, 网络 {l} 胜 / 老师 {w} 胜 / 和 {d}"
          f"  → 网络胜率 {score * 100:.0f}%  (已记入 {os.path.basename(args.log)})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
