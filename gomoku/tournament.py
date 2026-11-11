"""同门对战：用最新快照去挑战历史快照，按先后手分别统计。

设计要点
  * 两个引擎都是确定性 MCTS（temperature=0），不加随机开局会把同一盘棋重复 N 遍，
    所以每局先随机下 4 手（和 bench 的做法一致）。
  * 每个对手打 2×games 局：一半让"新模型"执黑，一半执白，结果分开记，
    这样能看出"先手优势"和"代际差距"各自有多大。
  * 多进程并行（VM 4 核），每个进程单线程 torch。

用法（在 gomoku 目录）：
    python3 tournament.py --sims 60 --games 8 --out tournament.csv
"""
import argparse
import csv
import glob
import os
import re
from multiprocessing import Pool

import numpy as np

TARGETS = [516, 657, 721, 820, 1158, 1363, 1477, 1608]
TAG = {516: "起点(混训转19×19)", 657: "12→8局/轮", 721: "上限170", 820: "采样24手",
       1158: "模拟120", 1363: "模拟120+τ0.25", 1477: "模拟90", 1608: "最终版"}


def snapshots():
    """挑出最接近 TARGETS 的 19x19 快照。"""
    items = []
    for f in glob.glob("checkpoints/archive/*.pt"):
        m = re.search(r"iter(\d+)_(9x9|19x19)_(\d{8})_(\d{4})\.pt", os.path.basename(f))
        if m:
            items.append((int(m.group(1)), m.group(2), f))
    items.sort()
    pool = [x for x in items if x[1] == "19x19"]
    out = []
    for t in TARGETS:
        if not pool:
            continue
        best = min(pool, key=lambda x: abs(x[0] - t))
        if best not in out:
            out.append(best)
    return out


def play(new_net, old_net, size, sims, opening, new_black, rng):
    """返回 1=新模型赢, 0=和, -1=旧模型赢。"""
    from game import GomokuGame
    from mcts import MCTS
    m_new, m_old = MCTS(new_net), MCTS(old_net)
    g = GomokuGame(size, 5, True)
    mid = size // 2
    box = [r * size + c for r in range(max(0, mid - 4), min(size, mid + 5))
           for c in range(max(0, mid - 4), min(size, mid + 5))]
    for _ in range(opening):
        legal = g.legal_moves()
        if not legal:
            break
        cand = [x for x in box if g.board.reshape(-1)[x] == 0]
        fin, win = g.step(int(rng.choice(cand if cand else legal)))
        if fin:
            return 1 if (win == 1) == new_black else (-1 if win else 0)
    while True:
        legal = g.legal_moves()
        if not legal:
            return 0
        me_black = (g.current_player == 1)
        m = m_new if (me_black == new_black) else m_old
        pi = m.get_policy(g, sims, 0.0, False)
        mv = int(pi.argmax())
        if g.board.reshape(-1)[mv] != 0:
            mv = int(legal[int(np.argmax(pi[legal]))])
        fin, win = g.step(mv)
        if fin:
            if win == 0:
                return 0
            return 1 if (win == 1) == new_black else -1


def job(args):
    (old_path, new_path, new_black, games, sims, size, seed) = args
    import torch
    torch.set_num_threads(1)
    from model import load_model
    new_net, old_net = load_model(new_path), load_model(old_path)
    new_net.eval(); old_net.eval()
    rng = np.random.default_rng(seed)
    w = l = d = 0
    for _ in range(games):
        r = play(new_net, old_net, size, sims, 4, new_black, rng)
        if r > 0:
            w += 1
        elif r < 0:
            l += 1
        else:
            d += 1
    return w, l, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=60)
    ap.add_argument("--games", type=int, default=8, help="每种先后手各打多少局")
    ap.add_argument("--size", type=int, default=19)
    ap.add_argument("--out", default="tournament.csv")
    ap.add_argument("--targets", default="", help="逗号分隔的轮数，如 527,820,1158")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()

    global TARGETS
    if a.targets:
        TARGETS = [int(x) for x in a.targets.split(",") if x.strip()]
    snaps = snapshots()
    new_iter, _, new_path = snaps[-1]
    print(f"新模型: iter{new_iter} ({new_path})", flush=True)
    print(f"对手: {[(it, TAG.get(it, '')) for it, _, _ in snaps[:-1]]}", flush=True)

    tasks = []
    for i, (it, _, path) in enumerate(snaps[:-1]):
        for k, black in enumerate((True, False)):
            tasks.append((path, new_path, black, a.games, a.sims, a.size,
                          1000 * i + k))

    with Pool(min(a.procs, len(tasks))) as pool:
        results = pool.map(job, tasks)

    rows, idx = [], 0
    for it, _, path in snaps[:-1]:
        for black in (True, False):
            w, l, d = results[idx]
            idx += 1
            rows.append({"对手轮数": it, "对手阶段": TAG.get(it, ""),
                         "新模型执": "黑" if black else "白",
                         "胜": w, "负": l, "和": d, "局数": w + l + d,
                         "胜率": round((w + 0.5 * d) / max(1, w + l + d), 3)})
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"已写出 {a.out}", flush=True)
    for r in rows:
        print(f"  vs iter{r['对手轮数']:<5} 新执{r['新模型执']}  "
              f"{r['胜']}胜 {r['负']}负 {r['和']}和  胜率 {r['胜率']:.0%}", flush=True)


if __name__ == "__main__":
    main()
