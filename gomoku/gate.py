"""代际闸门：让「最新权重」挑战「当前最佳」，决定是否替换、是否报警。

设计（训练与部署分离）
  * 训练永远写 checkpoints/latest.pt，保证模型一直在动；
  * 部署读 checkpoints/best.pt，保证线上不会退步；
  * 本脚本每 N 轮跑一次 40 局对抗（先手 20 / 后手 20，随机 4 手开局）：
        - 胜率 >= PROMOTE  → 把 latest 提升为 best（部署跟着换）
        - 胜率 <  50%      → 记一次"低迷"，连续 LOW_K 次则写 gate_alert.txt
        - 连续多次在 50%±SAT 内 → 判定饱和（SATURATED），写进状态文件
  * 只做"选择"，不冻结训练；漂移仍然可以继续。

用法：
    python3 gate.py                       # 跑一次评估
    python3 gate.py --games 20 --sims 120
"""
import argparse
import csv
import json
import os
import shutil
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "checkpoints")
LATEST = os.path.join(CKPT, "latest.pt")
BEST = os.path.join(CKPT, "best.pt")
STATE = os.path.join(CKPT, "gate_state.json")
LOG = os.path.join(CKPT, "gate_log.csv")
ALERT = os.path.join(CKPT, "gate_alert.txt")

PROMOTE = 0.55      # 达到这个胜率才替换最佳
LOW_K = 3           # 连续低于 50% 多少次报警
SAT_BAND = 0.05     # 长期卡在 50%±5% 视为饱和


def iter_of(path):
    import torch
    try:
        return torch.load(path, map_location="cpu").get("iter", -1)
    except Exception:
        return -1


def _play(new_path, best_path, new_black, games, sims, size, seed):
    import torch
    torch.set_num_threads(1)
    from game import GomokuGame
    from mcts import MCTS
    from model import load_model

    new_net, best_net = load_model(new_path), load_model(best_path)
    new_net.eval(); best_net.eval()
    m_new, m_best = MCTS(new_net), MCTS(best_net)
    rng = np.random.default_rng(seed)
    w = l = d = 0
    box = [r * size + c for r in range(size // 2 - 4, size // 2 + 5)
           for c in range(size // 2 - 4, size // 2 + 5)]
    for _ in range(games):
        g = GomokuGame(size, 5, True)
        for _ in range(4):                          # 随机开局打散确定性
            legal = g.legal_moves()
            if not legal:
                break
            cand = [x for x in box if g.board.reshape(-1)[x] == 0]
            fin, win = g.step(int(rng.choice(cand if cand else legal)))
            if fin:
                break
        result = 0
        while True:
            legal = g.legal_moves()
            if not legal:
                break
            if (g.current_player == 1) == new_black:
                pi = m_new.get_policy(g, sims, 0.0, False)
            else:
                pi = m_best.get_policy(g, sims, 0.0, False)
            mv = int(pi.argmax())
            if g.board.reshape(-1)[mv] != 0:
                mv = int(legal[int(np.argmax(pi[legal]))])
            fin, win = g.step(mv)
            if fin:
                result = 0 if win == 0 else (1 if (win == 1) == new_black else -1)
                break
        if result > 0:
            w += 1
        elif result < 0:
            l += 1
        else:
            d += 1
    return w, l, d


def engine_rate(path, games=24, sims=60, time_limit=0.1):
    """对传统引擎的胜率（外部基准，与自对弈胜率是两回事）。"""
    import subprocess
    import tempfile
    out = os.path.join(tempfile.gettempdir(), "gate_bench.csv")
    try:
        os.remove(out)
    except OSError:
        pass
    subprocess.run([os.sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "bench_vs_teacher.py"),
                    "--ckpt", path, "--games", str(games), "--size", "19",
                    "--sims", str(sims), "--time", str(time_limit),
                    "--open-moves", "4", "--log", out, "--quiet"],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
    try:
        import csv as _csv
        rows = list(_csv.DictReader(open(out, encoding="utf-8-sig")))
        if rows:
            return float(rows[-1]["网络胜率"])
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20, help="每种先后手各打多少局")
    ap.add_argument("--sims", type=int, default=120)
    ap.add_argument("--size", type=int, default=19)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--bench-games", type=int, default=48, help="提升前对传统引擎的验证局数（<48 局噪声太大）")
    ap.add_argument("--dir", default="", help="权重目录（默认 gomoku/checkpoints）")
    a = ap.parse_args()

    global LATEST, BEST, STATE, LOG, ALERT
    if a.dir:
        LATEST = os.path.join(a.dir, "latest.pt")
        BEST = os.path.join(a.dir, "best.pt")
        STATE = os.path.join(a.dir, "gate_state.json")
        LOG = os.path.join(a.dir, "gate_log.csv")
        ALERT = os.path.join(a.dir, "gate_alert.txt")
    if not os.path.exists(BEST):
        shutil.copy2(LATEST, BEST)
        print(f"[gate] 初始化 best.pt（复制自 latest.pt，iter {iter_of(LATEST)}）", flush=True)

    it_new, it_best = iter_of(LATEST), iter_of(BEST)
    if it_new <= it_best:
        print(f"[gate] latest({it_new}) 不比 best({it_best}) 新，跳过", flush=True)
        return

    # 4 个任务并行：2 种先后手 × 2 段
    half = max(1, a.games // 2)
    jobs = []
    for k, black in enumerate((True, False)):
        for s in range(2):
            jobs.append((LATEST, BEST, black, half, a.sims, a.size, 100 * k + s))
    if a.procs > 1:
        with Pool(min(a.procs, len(jobs))) as pool:
            parts = pool.starmap(_play, jobs)
    else:                                   # 单进程（沙箱里没有命名管道时用这个）
        parts = [_play(*j) for j in jobs]
    w = sum(p[0] for p in parts); l = sum(p[1] for p in parts); d = sum(p[2] for p in parts)
    n = w + l + d
    rate = (w + 0.5 * d) / max(1, n)

    state = {"best_iter": it_best, "low_streak": 0, "runs": []}
    if os.path.exists(STATE):
        try:
            state.update(json.load(open(STATE, encoding="utf-8")))
            state["best_iter"] = it_best
        except Exception:
            pass

    action = "KEEP"
    if rate >= PROMOTE:
        # 第二道检验：自对弈更强 ≠ 对传统引擎更强（1628 就是反例：61% vs 1477，但引擎只有 17%）
        best_ref = state.get("best_engine", 0.40)   # 128ch@2148 的 48 局基准
        cur = engine_rate(LATEST, games=a.bench_games, sims=60)   # 固定 60，和历史基准同口径
        print(f"[gate] 外部基准: 最新 {cur if cur is None else round(cur, 3)}"
              f" vs 最佳参考 {best_ref:.2f}", flush=True)
        if cur is None or cur >= best_ref - 0.07:   # 容差 7 个百分点（48 局噪声约 ±13%，取保守值）
            shutil.copy2(LATEST, BEST)
            state["best_iter"] = it_new
            state["best_engine"] = cur if cur is not None else best_ref
        else:
            action = "REJECT_ENGINE"
    if rate >= PROMOTE and action == "KEEP":
        state["low_streak"] = 0
        action = "PROMOTE"
        try:
            os.remove(ALERT)
        except OSError:
            pass
    elif rate < 0.50:
        state["low_streak"] = state.get("low_streak", 0) + 1
        action = f"LOW({state['low_streak']})"
        if state["low_streak"] >= LOW_K:
            with open(ALERT, "w", encoding="utf-8") as f:
                f.write(f"连续 {state['low_streak']} 次低于 50%\n"
                        f"latest iter {it_new} vs best {it_best}: {rate:.0%}\n"
                        f"时间 {time.strftime('%Y-%m-%d %H:%M')}\n")
            action = "ALERT"
    else:
        state["low_streak"] = 0

    runs = state.get("runs", [])
    runs.append(round(rate, 3))
    state["runs"] = runs[-20:]
    window = runs[-6:]
    if len(window) >= 5 and all(abs(x - 0.5) <= SAT_BAND for x in window):
        state["saturated"] = True
    else:
        state["saturated"] = False
    json.dump(state, open(STATE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    new_file = not os.path.exists(LOG)
    with open(LOG, "a", newline="", encoding="utf-8-sig" if new_file else "utf-8") as f:
        wr = csv.writer(f)
        if new_file:
            wr.writerow(["时间", "最新轮数", "最佳轮数", "局数", "胜", "负", "和",
                         "胜率", "动作", "连续低迷", "饱和"])
        wr.writerow([time.strftime("%Y-%m-%d %H:%M"), it_new, it_best, n, w, l, d,
                     f"{rate:.3f}", action, state.get("low_streak", 0),
                     state.get("saturated", False)])
    print(f"[gate] latest {it_new} vs best {it_best}: {w}胜 {l}负 {d}和 "
          f"= {rate:.0%} → {action}"
          f"{'  [饱和]' if state.get('saturated') else ''}", flush=True)


if __name__ == "__main__":
    main()
