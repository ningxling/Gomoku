"""用路线 1 老师给 kimi3 的网络做“监督预训练 + 老师陪练”，再交回 train.py 继续进化。

为什么有用
----------
kimi3 的 train.py 是纯 AlphaZero：从随机权重开始自我对弈。前几百轮网络几乎是乱下，
自我对弈学得非常慢（你现在这个 checkpoint 迭代 263 轮，策略基本还是均匀分布）。
传统引擎可以直接给出“人类/棋谱级”的着法，用它当老师做热启动，就把“从零学棋”
换成了“先模仿一个会下棋的老师，再自我提升”。

流水线
------
    老师自我对弈  ──►  (局面, 老师的软策略 pi, 胜负 z)  ──►  监督训练
                                                              │
                         （可选）网络自我对弈 / 老师陪练  ──►  │
                                                              ▼
                                            checkpoints/pretrain.pt
                                                              │
                                    train.py --ckpt ... 继续 AlphaZero
                                                              │
                                           duel.py 量化棋力变化

用法
----
    python pretrain.py                                   # 9x9, 60 局老师对局, 3 轮
    python pretrain.py --size 15 --teacher-games 200 --time 0.25 --epochs 4
    python pretrain.py --init latest                     # 在已有网络上继续蒸馏
    python pretrain.py --teacher-games 40 --selfplay-games 10   # 老师对局 + 网络自我对弈混合
    python duel.py --ckpt checkpoints/pretrain.pt --size 15 --games 10

产出 checkpoints/pretrain.pt 与 train.py / server.py 的格式完全相同，可以直接:
    copy checkpoints\\pretrain.pt checkpoints\\latest.pt
    python train.py
"""
import argparse
import os
import time

import numpy as np
import torch

from model import PolicyValueNet
from teacher import (NetAgent, TeacherAgent, augment, match, net_vs_teacher,
                     play_game, teacher_vs_teacher)


def load_net(path, channels, blocks, device):
    if not path or not os.path.exists(path):
        if path and path != "scratch":
            print(f"警告: 找不到 {path}, 用随机初始化")
        return PolicyValueNet(channels, blocks).to(device), 0
    ckpt = torch.load(path, map_location=device)
    a = ckpt.get("args", {})
    net = PolicyValueNet(a.get("channels", channels), a.get("blocks", blocks)).to(device)
    net.load_state_dict(ckpt["model"])
    print(f"已在 {path} 上继续（原训练轮数 {ckpt.get('iter', '?')}）")
    return net, int(ckpt.get("iter", 0) or 0)


def train_epochs(net, opt, data, epochs, batch_size, device, rng):
    net.train()
    n = len(data)
    for ep in range(epochs):
        order = rng.permutation(n)
        tot = pl = vl = 0.0
        steps = 0
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            if len(idx) < 2:
                continue
            states = torch.from_numpy(np.stack([data[j][0] for j in idx])).to(device)
            pis = torch.from_numpy(np.stack([data[j][1] for j in idx])).to(device)
            zs = torch.from_numpy(np.array([data[j][2] for j in idx],
                                           dtype=np.float32)).to(device)
            logits, v = net(states)
            p_loss = -(pis * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            v_loss = torch.nn.functional.mse_loss(v.squeeze(-1), zs)
            loss = p_loss + v_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            pl += p_loss.item()
            vl += v_loss.item()
            steps += 1
        if steps:
            print(f"  epoch {ep + 1}/{epochs}: loss {tot / steps:.3f} "
                  f"(policy {pl / steps:.3f} value {vl / steps:.3f})", flush=True)


@torch.no_grad()
def holdout_accuracy(net, data, device, limit=3000, batch=256):
    """在留出集上报告：策略 top-1 与老师一致率、价值 MSE。"""
    if not data:
        return None
    net.eval()
    data = data[:limit]
    hit = 0
    vse = 0.0
    for i in range(0, len(data), batch):
        chunk = data[i:i + batch]
        states = torch.from_numpy(np.stack([d[0] for d in chunk])).to(device)
        logits, v = net(states)
        pred = logits.argmax(dim=1).cpu().numpy()
        tgt = np.stack([d[1] for d in chunk])
        hit += int((pred == tgt.argmax(axis=1)).sum())
        z = np.array([d[2] for d in chunk], dtype=np.float32)
        vse += float(((v.squeeze(-1).cpu().numpy() - z) ** 2).sum())
    return hit / len(data), vse / len(data)


def main():
    ap = argparse.ArgumentParser(description="用路线1老师预训练 kimi3 的网络")
    ap.add_argument("--size", type=int, default=9)
    ap.add_argument("--teacher-games", type=int, default=60)
    ap.add_argument("--selfplay-games", type=int, default=0,
                    help="额外加入网络自我对弈局数（老师陪练/自我提升）")
    ap.add_argument("--vs-teacher-games", type=int, default=0,
                    help="网络和老师直接对打的局数（专家迭代，最接近“继续强化训练”）")
    ap.add_argument("--time", type=float, default=0.2, help="老师每步思考秒数")
    ap.add_argument("--temperature", type=float, default=0.3, help="软标签温度")
    ap.add_argument("--topk", type=int, default=4, help="软标签保留的候选着法数")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rounds", type=int, default=1, help="重复多少轮“生成数据+训练”")
    ap.add_argument("--init", default="scratch", help="scratch / latest / 权重路径")
    ap.add_argument("--ckpt", default="checkpoints/pretrain.pt")
    ap.add_argument("--no-forbidden", action="store_true")
    ap.add_argument("--no-augment", action="store_true", help="关闭 8 对称数据增强")
    ap.add_argument("--argmax-moves", action="store_true", help="老师对局用 argmax 而非采样")
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--eval-games", type=int, default=0, help="训练后与老师对打的局数")
    ap.add_argument("--sims", type=int, default=100, help="评估/自我对弈时网络的模拟次数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else \
        ("cpu" if args.device == "auto" else args.device)
    forbidden = not args.no_forbidden
    init_path = None
    if args.init == "latest":
        init_path = "checkpoints/latest.pt"
    elif args.init != "scratch":
        init_path = args.init

    print(f"设备 {device} | 棋盘 {args.size}x{args.size} | 禁手 {'开' if forbidden else '关'} | "
          f"老师 {args.time}s/手 | 初始化 {args.init}", flush=True)
    net, base_iter = load_net(init_path, args.channels, args.blocks, device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    teacher = TeacherAgent(time_limit=args.time, forbidden=forbidden,
                           temperature=args.temperature, topk=args.topk)

    data = []
    for rd in range(1, args.rounds + 1):
        t0 = time.time()
        print(f"\n=== 第 {rd}/{args.rounds} 轮: 生成数据 ===", flush=True)
        new = teacher_vs_teacher(
            games=args.teacher_games, size=args.size, forbidden=forbidden,
            time_limit=args.time, temperature=args.temperature, topk=args.topk,
            seed=args.seed + rd * 977, sample_moves=not args.argmax_moves,
            sym_aug=not args.no_augment)

        if args.selfplay_games > 0:
            student = NetAgent(net, sims=args.sims, temperature=1.0)
            extra = []
            for g in range(args.selfplay_games):
                samples, winner, nm = play_game({1: student.policy, -1: student.policy},
                                                args.size, 5, forbidden,
                                                sample=True, rng=rng)
                extra.extend(augment(samples, args.size, not args.no_augment))
            print(f"  网络自我对弈 {args.selfplay_games} 局，样本 {len(extra)}")
            new.extend(extra)

        if args.vs_teacher_games > 0:
            print(f"--- 老师陪练 {args.vs_teacher_games} 局 ---", flush=True)
            student = NetAgent(net, sims=args.sims, temperature=0.0)
            extra, st = net_vs_teacher(args.vs_teacher_games, args.size, 5, forbidden,
                                       teacher=teacher, net_agent=student,
                                       seed=args.seed + rd * 131)
            print(f"  网络 {st['net']} 胜 / 老师 {st['teacher']} 胜 / 和 {st['draw']}，"
                  f"样本 {len(extra)}")
            new.extend(extra)

        data.extend(new)
        print(f"  本轮新增样本 {len(new)}，累计 {len(data)}，用时 {time.time() - t0:.0f}s",
              flush=True)

        rng.shuffle(data)
        cut = int(len(data) * (1 - args.val_split)) if args.val_split > 0 else len(data)
        train_set, val_set = data[:cut], data[cut:]

        print(f"--- 训练 {args.epochs} 轮（训练集 {len(train_set)}）---", flush=True)
        train_epochs(net, opt, train_set, args.epochs, args.batch_size, device, rng)

        acc = holdout_accuracy(net, val_set, device)
        if acc:
            print(f"  留出集: 策略 top-1 与老师一致率 {acc[0] * 100:.1f}%, "
                  f"价值 MSE {acc[1]:.3f}", flush=True)

        os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
        torch.save({"model": net.state_dict(),
                    "args": {"channels": args.channels, "blocks": args.blocks},
                    "size": args.size, "iter": base_iter + rd,
                    "teacher": {"games": args.teacher_games, "time": args.time,
                                "forbidden": forbidden, "round": rd,
                                "vs_teacher": args.vs_teacher_games,
                                "selfplay": args.selfplay_games}},
                   args.ckpt)
        print(f"  已保存 {args.ckpt}（轮数记为 {base_iter + rd}）", flush=True)

    if args.eval_games > 0:
        print(f"\n=== 与老师对打 {args.eval_games} 局 ===", flush=True)
        student = NetAgent(net, sims=args.sims, temperature=0.0)
        w, l, d = match(teacher.policy, student.policy, args.eval_games, args.size,
                        forbidden=forbidden, tag_a="老师", tag_b="网络")
        print(f"老师 {w} 胜 / 网络 {l} 胜 / 和 {d}")

    print("\n完成。接下来可以直接让 AlphaZero 从这份权重继续进化：")
    print(f"    copy {args.ckpt} checkpoints\\latest.pt")
    print("    python train.py            # 或者 python train.py --ckpt "
          f"{args.ckpt}")


if __name__ == "__main__":
    main()
