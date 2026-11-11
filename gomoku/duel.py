"""老师（路线 1 传统引擎） vs 学生（kimi3 的神经网络 + MCTS）对打。

用途：衡量网络现在到底有多强、预训练/自我对弈之后有没有变强。

例：
    python duel.py --size 9  --games 10 --sims 100          # 9x9, 每步 100 次模拟
    python duel.py --size 19 --games 6  --sims 200 --time 1
    python duel.py --scratch                                 # 随机权重，看下限
    python duel.py --ckpt checkpoints/pretrain.pt --against checkpoints/latest.pt
                                                             # 新权重 vs 老权重

输出：胜负和、胜率、等效 Elo 差。
"""
import argparse
import time

import numpy as np
import torch

from model import PolicyValueNet, load_model
from teacher import NetAgent, TeacherAgent, match, score_to_elo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/latest.pt")
    ap.add_argument("--against", default=None,
                    help="指定另一个网络权重：改成网络 A vs 网络 B")
    ap.add_argument("--scratch", action="store_true", help="不加载权重，用随机初始化的网络")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--size", type=int, default=9)
    ap.add_argument("--sims", type=int, default=100, help="网络每步 MCTS 模拟次数")
    ap.add_argument("--time", type=float, default=0.5, help="老师每步思考秒数")
    ap.add_argument("--no-forbidden", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.scratch:
        net = PolicyValueNet().to(device)
        print("使用随机初始化的网络（未训练）", flush=True)
    else:
        net = load_model(args.ckpt, device=device)
        print(f"已加载网络权重 {args.ckpt}", flush=True)
    net.eval()
    student = NetAgent(net, sims=args.sims, temperature=0.0)

    if args.against:
        net2 = load_model(args.against, device=device)
        net2.eval()
        other = NetAgent(net2, sims=args.sims, temperature=0.0)
        print(f"对比对象: {args.against}；{args.games} 局, {args.size}x{args.size}", flush=True)
        t0 = time.time()
        wins, losses, draws = match(student.policy, other.policy, args.games, args.size,
                                    forbidden=not args.no_forbidden,
                                    tag_a="A", tag_b="B")
        n = max(1, wins + losses + draws)
        score = (wins + 0.5 * draws) / n
        print(f"\n结果: A({args.ckpt}) {wins} 胜 / B({args.against}) {losses} 胜 / 和 {draws}"
              f"  (共 {n} 局, {time.time() - t0:.0f}s)")
        print(f"A 胜率 {score * 100:.0f}%,  等效 Elo 差 {score_to_elo(score):+.0f}")
        return

    teacher = TeacherAgent(time_limit=args.time, forbidden=not args.no_forbidden)
    print(f"对局: {args.games} 局, {args.size}x{args.size}, "
          f"禁手 {'开' if not args.no_forbidden else '关'}, "
          f"老师 {args.time}s/手, 网络 {args.sims} 次模拟", flush=True)
    t0 = time.time()
    wins, losses, draws = match(teacher.policy, student.policy, args.games, args.size,
                                forbidden=not args.no_forbidden,
                                tag_a="老师", tag_b="网络")
    n = max(1, wins + losses + draws)
    score = (wins + 0.5 * draws) / n
    print(f"\n结果: 老师 {wins} 胜 / 网络 {losses} 胜 / 和 {draws}  "
          f"(共 {n} 局, {time.time() - t0:.0f}s)")
    print(f"老师胜率 {score * 100:.0f}%,  等效 Elo 差 {score_to_elo(score):+.0f}")


if __name__ == "__main__":
    main()
