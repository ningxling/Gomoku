"""自我对弈训练脚本 (AlphaZero 简化版, 支持课程式棋盘扩大)。

用法:
    python train.py                          # 默认课程: 9 -> 13 -> 15 -> 19
    python train.py --curriculum "15:0"      # 固定 15x15 训练
    python train.py --no-forbidden           # 关闭禁手规则

课程式训练: --curriculum "9:0,13:30,15:100,19:250" 表示
第 0~29 轮用 9x9, 第 30~99 轮用 13x13, 第 100~249 轮用 15x15, 之后用 19x19。
网络是全卷积的, 权重与棋盘尺寸无关, 换尺寸无需重新训练。

Ctrl+C 可随时停止, 最新权重保存在 checkpoints/latest.pt, 再启动会自动续训。
"""
import argparse
import math
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from game import GomokuGame
from mcts import MCTS
from model import PolicyValueNet


def parse_curriculum(text):
    stages = []
    for part in text.split(","):
        size, start = part.split(":")
        stages.append((int(start), int(size)))
    stages.sort()
    return [s[1] for s in stages], [s[0] for s in stages]


def curriculum_size(iteration, sizes, starts):
    size = sizes[0]
    for s, start in zip(sizes, starts):
        if iteration >= start:
            size = s
    return size


def parse_mix(text):
    """'9:1,19:2' -> ([9, 19], [0.333, 0.667])，按权重每轮随机选棋盘大小。"""
    sizes, weights = [], []
    for part in text.split(","):
        s, w = part.split(":")
        sizes.append(int(s))
        weights.append(float(w))
    total = sum(weights)
    return sizes, [w / total for w in weights]


def _cells_of(game):
    """kimi3 的 board(1/-1) -> 传统引擎的 1/2 一维棋盘。"""
    n = game.size
    out = [0] * (n * n)
    b = game.board
    for r in range(n):
        row = b[r]
        for c in range(n):
            v = int(row[c])
            if v == 1:
                out[r * n + c] = 1
            elif v == -1:
                out[r * n + c] = 2
    return out


class TacticalGuard:
    """根节点战术护栏（第一档强制战术扩展）。

    每一步先用传统引擎的威胁表查一次：
      * 我方有冲四点（下一手成五）→ 直接走，不用 MCTS；
      * 对方只有唯一一个冲四点 → 只能挡，也不用 MCTS；
      * 其余情况交回 MCTS。
    这样自对弈的 π 在这些位置是精确的，而且对局不会在 18 手就崩掉。
    """

    def __init__(self, forbidden=True):
        from engine import Engine
        self.eng = Engine(time_limit=0.05, forbidden=forbidden)
        self.hits = 0
        self.moves = 0

    def reset(self, game):
        self.eng.set_position(game.size, _cells_of(game),
                              1 if game.current_player == 1 else 2)

    def forced(self, game):
        me = 1 if game.current_player == 1 else 2
        legal = set(game.legal_moves())
        mine = [c for c in self.eng.tcnt[me] if c in legal]
        if mine:
            self.hits += 1
            return mine[0]
        theirs = [c for c in self.eng.tcnt[3 - me] if c in legal]
        if len(theirs) == 1:            # 双四就挡不住了，交给 MCTS 吃这个 -1
            self.hits += 1
            return theirs[0]
        return None

    def apply(self, move, player):
        self.moves += 1
        self.eng._make(move, 1 if player == 1 else 2)


def _bootstrap_value(game, guard):
    """截断对局的“真值”不能标 0（那是假和棋）。用传统引擎在截断局面上的判断当目标：

    - 引擎搜索到强制胜 -> score 极大 -> tanh 后约 ±1；
    - 否则给一个基于棋型的估值，tanh 压到 (-1, 1)。

    这是独立信号（不依赖网络自己的价值头），所以不会掉进“所有局面都是 0”的固定点。
    """
    try:
        if guard is not None:
            eng = guard.eng
        else:
            from engine import Engine
            eng = Engine(time_limit=0.05, forbidden=game.use_forbidden)
            eng.set_position(game.size, _cells_of(game),
                             1 if game.current_player == 1 else 2)
        _mv, info = eng.choose_move()
        return math.tanh(float(info.get("score", 0)) / 1500.0)
    except Exception:                                   # noqa: BLE001
        return 0.0


def self_play_episode(net, mcts, size, win_len, sims, temp_moves, use_forbidden,
                      guard=None, max_moves=0, temp=1.0):
    """打一局自我对弈，返回 ([(编码局面, MCTS策略, 最终价值z), ...], 结果, bootstrap值)。

    结果取值: "black" / "white" / "draw"（双方无子可下）/ "trunc"（超过 max_moves 手截断）。
    截断局用传统引擎的估值做 bootstrap，而不是标成和棋。
    """
    game = GomokuGame(size, win_len, use_forbidden)
    history = []
    if guard is not None:
        guard.reset(game)
    outcome = "draw"
    while True:
        legal = game.legal_moves()
        if not legal:
            break
        forced = guard.forced(game) if guard is not None else None
        if forced is not None:
            # 战术护栏命中：π 直接是精确的 one-hot，不跑 MCTS
            pi = np.zeros(size * size, dtype=np.float64)
            pi[forced] = 1.0
        else:
            pi = mcts.get_policy(game, sims, temperature=temp, add_noise=True)
        history.append((game.encode(), pi, game.current_player))
        if len(history) <= temp_moves:
            move = int(np.random.choice(len(pi), p=pi))
        else:
            move = int(pi.argmax())
        if game.board.reshape(-1)[move] != 0:       # 兜底：pi 万一有异常也绝不下到已占点
            move = int(legal[int(np.argmax(pi[legal]))])
        player = game.current_player
        finished, winner = game.step(move)
        if guard is not None:
            guard.apply(move, player)
        if finished:
            outcome = "draw" if winner == 0 else ("black" if winner == 1 else "white")
            break
        if max_moves and len(history) >= max_moves:
            outcome = "trunc"
            break
    if outcome == "trunc":
        # 用传统引擎对截断局面的判断做 bootstrap（不是假和棋 0）
        v = _bootstrap_value(game, guard)
        to_move = game.current_player
        zs = [v if pl == to_move else -v for _, _, pl in history]
    elif outcome == "draw":
        zs = [0.0] * len(history)
        v = 0.0
    else:
        w = 1 if outcome == "black" else -1
        zs = [1.0 if pl == w else -1.0 for _, _, pl in history]
        v = None
    return [(s, p, z) for (s, p, _), z in zip(history, zs)], outcome, v


def relabel_live_threes(buf, size, forbidden, k, time_limit):
    """定向标注：抽 k 个局面，凡"对手有活三"的，用传统引擎的着法覆盖策略标签 π。

    为什么这么做：干净局面下模型本来就会挡活三（构造测试全对），
    但在真实对局的复杂局面里会漏——用模型自己会遇到的局面做硬样本，
    把"该挡哪"的标签喂回去，比继续堆自我对弈更能补这个洞。
    """
    from tactics import live_three_blocks
    try:
        from engine import Engine
    except Exception:
        return 0
    if not buf:
        return 0
    items = list(buf)
    idxs = np.random.choice(len(items), min(k, len(items)), replace=False)
    eng = Engine(time_limit=time_limit, forbidden=forbidden)
    hit = 0
    for i in idxs:
        s, p, z = items[i]
        me = 1 if float(s[2, 0, 0]) > 0.5 else -1
        board = (s[0].astype(np.float64) - s[1].astype(np.float64)) * me
        found, _blocks = live_three_blocks(board, me, size)
        if not found:
            continue
        cells = [0 if v == 0 else (1 if v > 0 else 2) for v in board.ravel()]
        eng.forbidden = forbidden
        eng.set_position(size, cells, 1 if me == 1 else 2)
        mv, _info = eng.choose_move()
        if mv is None:
            continue
        pi = np.zeros(size * size, dtype=np.float32)
        pi[mv] = 1.0
        items[i] = (s, pi, z)
        hit += 1
    if hit:
        buf.clear()
        buf.extend(items)
    return hit


def spar_vs_engine(net, mcts, engine, size, win_len, forbidden, sims, rng, games):
    """混合对抗训练：模型 vs 传统引擎。

    状态来自**模型自己实际走出来的局面**（会踩进它的盲区），
    动作标签用**引擎的着法**（DAgger 式），结果用真实胜负。
    这样梯度里才第一次出现"你在这里不挡，然后就输了"的信号——
    纯自我对弈里两个同源模型共享盲区，永远给不出这个信号。
    """
    from game import GomokuGame
    data = []
    for gi in range(games):
        model_side = 1 if gi % 2 == 0 else -1
        g = GomokuGame(size, win_len, forbidden)
        for _ in range(4):                          # 随机开局
            legal = g.legal_moves()
            if not legal:
                break
            fin, _w = g.step(int(rng.choice(legal)))
            if fin:
                break
        recs, winner = [], 0
        while True:
            legal = g.legal_moves()
            if not legal:
                break
            me = g.current_player
            cells = [0 if v == 0 else (1 if v > 0 else 2) for v in g.board.ravel()]
            engine.forbidden = forbidden
            engine.set_position(size, cells, 1 if me == 1 else 2)
            mv_exp, _info = engine.choose_move()
            if mv_exp is None:
                break
            if me == model_side:                    # 轮到模型：记样本，模型自己走
                pi = np.zeros(size * size, dtype=np.float32)
                pi[mv_exp] = 1.0
                recs.append((g.encode(), pi, me))
                pim = mcts.get_policy(g, sims, 0.0, False)
                mv = int(np.argmax(pim))
                if g.board.reshape(-1)[mv] != 0:
                    mv = int(legal[int(np.argmax(pim[legal]))])
            else:                                   # 轮到引擎
                mv = int(mv_exp)
            fin, w = g.step(mv)
            if fin:
                winner = w
                break
        for s, p, me in recs:
            z = 0.0 if winner == 0 else (1.0 if winner == me else -1.0)
            data.append((s, p, z))
    return data


def train_step(net, optimizer, batch, device):
    states = torch.from_numpy(np.stack(batch[0])).to(device)
    target_pi = torch.from_numpy(np.stack(batch[1])).to(device)
    target_v = torch.from_numpy(np.array(batch[2], dtype=np.float32)).to(device)
    logits, v = net(states)
    v_loss = F.mse_loss(v.squeeze(-1), target_v)
    p_loss = -(target_pi * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    loss = v_loss + p_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), p_loss.item(), v_loss.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curriculum", default="9:0,13:30,15:100,19:250",
                    help="课程表 '尺寸:起始轮数,...'，如 9:0,13:30,15:100,19:250")
    ap.add_argument("--win-len", type=int, default=5)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--sims", type=int, default=100, help="每步 MCTS 模拟次数")
    ap.add_argument("--episodes", type=int, default=8, help="每个 iteration 自我对弈局数")
    ap.add_argument("--iters", type=int, default=100000)
    ap.add_argument("--temp-moves", type=int, default=8, help="前多少步用温度采样保证探索")
    ap.add_argument("--buffer", type=int, default=30000, help="每种尺寸的经验池容量")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=5, help="每个 iteration 训练轮数")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-decay-every", type=int, default=250,
                    help="每多少轮将学习率乘以衰减系数, 0 表示不衰减")
    ap.add_argument("--lr-decay-factor", type=float, default=0.5)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--no-forbidden", action="store_true", help="关闭黑棋禁手规则")
    ap.add_argument("--teacher-games", type=int, default=0,
                    help="每轮额外加入多少局“老师(传统 α-β 引擎)”对局做示范数据，0=纯自我对弈")
    ap.add_argument("--teacher-time", type=float, default=0.12, help="老师每步思考秒数")
    ap.add_argument("--teacher-temp", type=float, default=0.3, help="老师软标签温度")
    ap.add_argument("--mix-sizes", default="",
                    help="按权重随机选棋盘大小，如 '9:1,19:2'；设了就忽略 --curriculum")
    ap.add_argument("--tactical-guard", type=int, default=1,
                    help="自我对弈是否启用根节点战术护栏（能成五/必挡四直接走），1=开 0=关")
    ap.add_argument("--max-moves", type=int, default=0,
                    help="自我对弈单局手数上限，超过按和棋截断（0=不限）")
    ap.add_argument("--spar-games", type=int, default=0,
                    help="每轮和传统引擎下的对抗局数（0=关闭；每局约 3~5 秒）")
    ap.add_argument("--spar-ratio", type=float, default=0.25,
                    help="每批训练样本里对抗样本的占比")
    ap.add_argument("--tactical-mix", type=int, default=0,
                    help="每轮从经验池抽多少个局面做活三定向标注（0=关闭，建议 60）")
    ap.add_argument("--temp", type=float, default=0.25,
                    help="采样温度：1.0 太噪（对局 10 手就崩），0.25 既有多样性又不乱下")
    ap.add_argument("--ckpt", default="checkpoints/latest.pt")
    args = ap.parse_args()

    mix_sizes, mix_p = parse_mix(args.mix_sizes) if args.mix_sizes else (None, None)

    sizes, starts = parse_curriculum(args.curriculum)
    use_forbidden = not args.no_forbidden
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device} | 课程: {args.curriculum} | 禁手: {use_forbidden} | "
          f"每步模拟: {args.sims}", flush=True)

    net = PolicyValueNet(args.channels, args.blocks).to(device)
    start_iter = 1
    if os.path.exists(args.ckpt):
        saved = torch.load(args.ckpt, map_location=device)
        net.load_state_dict(saved["model"])
        start_iter = saved.get("iter", 0) + 1
        print(f"已加载已有权重: {args.ckpt} (从第 {start_iter} 轮继续训练)", flush=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    mcts = MCTS(net)
    guard = None
    if args.tactical_guard:
        try:
            guard = TacticalGuard(use_forbidden)
            print("已启用根节点战术护栏（成五/唯一挡四直接走，不跑 MCTS）", flush=True)
        except Exception as e:                       # 没有 engine.py 就退回纯自我对弈
            print(f"战术护栏不可用({type(e).__name__}: {e})，退回纯自我对弈", flush=True)
    buffers = {}  # 不同尺寸的局面临界形状不同, 分尺寸存经验池
    last_size = None
    spar_buf = deque(maxlen=3000)      # 对抗陪练样本（单独存放，按比例混入批次）
    spar_engine = None
    if args.spar_games > 0:
        try:
            from engine import Engine
            spar_engine = Engine(time_limit=args.teacher_time, forbidden=use_forbidden)
            print(f"已启用混合对抗训练：每轮 {args.spar_games} 局 vs 传统引擎，"
                  f"批次中对抗样本占 {args.spar_ratio:.0%}", flush=True)
        except Exception as e:
            print(f"对抗陪练不可用({type(e).__name__}: {e})，退回纯自我对弈", flush=True)

    for it in range(start_iter, args.iters + 1):
        t0 = time.time()
        if (args.lr_decay_every and it > start_iter
                and (it - start_iter) % args.lr_decay_every == 0):
            for g in optimizer.param_groups:
                g["lr"] = max(args.min_lr, g["lr"] * args.lr_decay_factor)
            print(f"=== 学习率调整为 {optimizer.param_groups[0]['lr']:.1e} ===", flush=True)
        if mix_sizes:
            size = int(np.random.choice(mix_sizes, p=mix_p))
        else:
            size = curriculum_size(it, sizes, starts)
            if size != last_size:
                print(f"=== 第 {it} 轮起切换到 {size}x{size} 棋盘 ===", flush=True)
                last_size = size
        buf = buffers.setdefault(size, deque(maxlen=args.buffer))

        game = GomokuGame(size, args.win_len, use_forbidden)  # 仅用于对称变换
        if guard is not None:
            guard.hits = 0
            guard.moves = 0
        res = {"black": 0, "white": 0, "draw": 0, "trunc": 0}
        plies = 0
        boots = []
        for _ in range(args.episodes):
            # 每局上限带点抖动：否则偶数手的 cap 会让“截断时轮到的一方”永远是同一种颜色，
            # 75% 的样本都变成“某色必胜”
            cap = int(args.max_moves * np.random.uniform(0.7, 1.0)) if args.max_moves else 0
            samples, outcome, boot = self_play_episode(
                net, mcts, size, args.win_len, args.sims, args.temp_moves,
                use_forbidden, guard=guard, max_moves=cap, temp=args.temp)
            res[outcome] = res.get(outcome, 0) + 1
            plies += len(samples)
            if outcome == "trunc" and boot is not None:
                boots.append(boot)
            for s, p, z in samples:
                # 8 种旋转/翻转 × 2（原色 / 交换黑白）= 16 倍增强
                # 交换黑白：己方与对方通道互换、先后手标识翻转；着法 p 与结果 z 不变
                for s_aug, p_aug in game.symmetries(s, p):
                    buf.append((s_aug, p_aug.astype(np.float32), z))
                    s_swap = s_aug[[1, 0, 2]].copy()
                    s_swap[2] = 1.0 - s_swap[2]
                    buf.append((s_swap, p_aug.astype(np.float32), z))
        boot_txt = (f" | 截断 bootstrap 均值 {sum(boots) / len(boots):+.2f}"
                    f" (|v|>=0.3 占 {sum(1 for b in boots if abs(b) >= 0.3) * 100 // len(boots)}%)"
                    if boots else "")
        print(f"  对局: 黑胜 {res['black']} / 白胜 {res['white']} / 和 {res['draw']}"
              f" / 截断 {res['trunc']} | 平均 {plies // max(1, args.episodes)} 手/局{boot_txt}",
              flush=True)
        if guard is not None and guard.moves:
            print(f"  战术护栏命中 {guard.hits}/{guard.moves} 手 "
                  f"({guard.hits * 100 // guard.moves}%)", flush=True)

        # 老师示范：用路线1传统引擎再下一轮对局，把样本混进同一个经验池
        if args.teacher_games > 0:
            try:
                from teacher import teacher_vs_teacher
                tdata = teacher_vs_teacher(
                    games=args.teacher_games, size=size, win_len=args.win_len,
                    forbidden=use_forbidden, time_limit=args.teacher_time,
                    temperature=args.teacher_temp, topk=4,
                    seed=(it * 7919) % 100000, sample_moves=True,
                    sym_aug=True, verbose=False)
                for s, p, z in tdata:
                    buf.append((s, p, z))
                print(f"  老师示范 {args.teacher_games} 局 -> {len(tdata)} 样本", flush=True)
            except Exception as e:                       # 老师不可用就退回纯自我对弈
                print(f"  老师不可用({type(e).__name__}: {e})，本轮跳过", flush=True)

        if args.tactical_mix > 0:
            n_hit = relabel_live_threes(buf, size, use_forbidden,
                                        args.tactical_mix, args.teacher_time)
            print(f"  定向标注: 抽 {args.tactical_mix} 局面，命中活三 {n_hit} 个"
                  f"（已用引擎着法覆盖标签）", flush=True)

        if spar_engine is not None:
            try:
                sdata = spar_vs_engine(net, mcts, spar_engine, size, args.win_len,
                                       use_forbidden, args.sims,
                                       np.random.default_rng(it * 7717), args.spar_games)
                for s_, p_, z_ in sdata:
                    for s_aug, p_aug in game.symmetries(s_, p_):
                        spar_buf.append((s_aug, p_aug.astype(np.float32), z_))
                wins = sum(1 for _s, _p, z_ in sdata if z_ > 0)
                total_g = max(1, sum(1 for _s, _p, _z in sdata))
                print(f"  对抗陪练 {args.spar_games} 局 -> {len(sdata)} 样本"
                      f"（模型位置胜率 {wins}/{len(sdata)}，库存 {len(spar_buf)}）", flush=True)
            except Exception as e:
                print(f"  对抗陪练出错({type(e).__name__}: {e})，本轮跳过", flush=True)

        net.train()
        losses = []
        for _ in range(args.epochs):
            # 按各经验池存量比例随机抽一个尺寸的批次 (旧尺寸经验不浪费)
            pools = [b for b in buffers.values() if len(b) >= 16]
            weights = np.array([len(b) for b in pools], dtype=np.float64)
            pool = pools[np.random.choice(len(pools), p=weights / weights.sum())]
            n_spar = 0
            if spar_engine is not None and len(spar_buf) >= 200:
                n_spar = min(int(args.batch_size * args.spar_ratio), len(spar_buf))
            idx = np.random.choice(len(pool), min(args.batch_size - n_spar, len(pool)),
                                   replace=False)
            items = [pool[i] for i in idx]
            if n_spar:
                sidx = np.random.choice(len(spar_buf), n_spar, replace=False)
                items += [spar_buf[i] for i in sidx]
            batch = list(zip(*items))
            losses.append(train_step(net, optimizer, batch, device))

        os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
        torch.save({"model": net.state_dict(),
                    "args": {"channels": args.channels, "blocks": args.blocks},
                    "size": size, "iter": it}, args.ckpt)
        l, pl, vl = np.mean(losses, axis=0)
        total = sum(len(b) for b in buffers.values())
        print(f"iter {it:4d} | {size}x{size} | 样本 {total:7d} | loss {l:.3f} "
              f"(policy {pl:.3f} value {vl:.3f}) | {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
