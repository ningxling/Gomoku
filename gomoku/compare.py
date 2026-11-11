"""成对比较：候选权重 vs 基线权重，判断"是否真的变强"。

设计要点
  * **成对开局**：每个开局下两盘（黑白互换），把"开局差异"从方差里消掉；
  * **bootstrap 置信区间**：对"每个开局的得分差"重采样，给出 95% CI；
  * **结论三分**：显著更好 / 显著更差 / 无法区分（不要拿噪声当进步）。

用法：
    python compare.py --base checkpoints/best.pt --cand checkpoints/latest.pt --pairs 20
    python compare.py --base A.pt --cand B.pt --pairs 20 --sims 120 --size 19
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np


def _play_pair(args):
    """一个开局的完整配对：候选执黑一盘 + 候选执白一盘。"""
    base_path, cand_path, seed, sims, size = args
    import torch
    torch.set_num_threads(1)
    from game import GomokuGame
    from mcts import MCTS
    from model import load_model

    base, cand = load_model(base_path), load_model(cand_path)
    base.eval(); cand.eval()
    m_base, m_cand = MCTS(base), MCTS(cand)
    rng = np.random.default_rng(seed)
    box = [r * size + c for r in range(size // 2 - 4, size // 2 + 5)
           for c in range(size // 2 - 4, size // 2 + 5)]

    def one(cand_black):
        g = GomokuGame(size, 5, True)
        for _ in range(4):                      # 每个开局先用同一个 seed 打散
            legal = g.legal_moves()
            if not legal:
                break
            c_ = [x for x in box if g.board.reshape(-1)[x] == 0]
            fin, win = g.step(int(rng.choice(c_ if c_ else legal)))
            if fin:
                break
        while True:
            legal = g.legal_moves()
            if not legal:
                return 0.5
            m = m_cand if ((g.current_player == 1) == cand_black) else m_base
            pi = m.get_policy(g, sims, 0.0, False)
            mv = int(pi.argmax())
            if g.board.reshape(-1)[mv] != 0:
                mv = int(legal[int(np.argmax(pi[legal]))])
            fin, win = g.step(mv)
            if fin:
                if win == 0:
                    return 0.5
                return 1.0 if (win == 1) == cand_black else 0.0

    r_black = one(True)
    r_white = one(False)
    return (r_black + r_white) / 2, r_black, r_white


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join("checkpoints", "best.pt"))
    ap.add_argument("--cand", default=os.path.join("checkpoints", "latest.pt"))
    ap.add_argument("--pairs", type=int, default=20, help="开局数（每局下两盘）")
    ap.add_argument("--sims", type=int, default=120)
    ap.add_argument("--size", type=int, default=19)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    def it_of(p):
        import torch
        return torch.load(p, map_location="cpu").get("iter", "?")

    t0 = time.time()
    jobs = [(a.base, a.cand, a.seed + i, a.sims, a.size) for i in range(a.pairs)]
    if a.procs > 1:
        with Pool(min(a.procs, len(jobs))) as pool:
            res = pool.map(_play_pair, jobs)
    else:
        res = [_play_pair(j) for j in jobs]

    per_pair = np.array([r[0] for r in res])
    n_games = 2 * len(res)
    score = per_pair.mean()

    rng = np.random.default_rng(12345)
    boots = [rng.choice(per_pair, size=len(per_pair), replace=True).mean()
             for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])

    print(f"基线 {it_of(a.base)}  vs  候选 {it_of(a.cand)}")
    print(f"开局 {len(res)} 个 / 对局 {n_games} 盘，每步模拟 {a.sims}，用时 {time.time()-t0:.0f}s")
    print(f"候选得分率 {score*100:.1f}%  95% CI [{lo*100:.1f}%, {hi*100:.1f}%]")
    if lo > 0.5:
        print("结论：候选**显著更好**（置信区间下界 >50%）")
    elif hi < 0.5:
        print("结论：候选**显著更差**（置信区间上界 <50%）")
    else:
        print("结论：**无法区分**——需要更多局数，别把噪声当进步")


if __name__ == "__main__":
    main()
