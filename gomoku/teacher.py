"""教师接口：把路线 1 引擎包装成 kimi3 训练管线能直接用的“老师”。

kimi3 的训练管线吃的是 (局面编码, 策略 pi, 结果 z) 三元组（见 train.py 的
self_play_episode）。本模块用传统引擎生成同样格式的示范数据，于是：

    * pretrain.py   : 用老师对局做监督预训练（热启动），让网络从“会下棋”开始，
                      而不是随机权重 —— 这是最划算的“辅助训练”；
    * duel.py       : 让网络和老师对打，量化棋力变化；
    * train.py      : 预训练完直接把 checkpoints/pretrain.pt 交给它继续自我对弈。

设计要点
--------
* 老师的着法直接来自 α-β 搜索，天然带“根节点各着法的分值”，
  root_policy() 能转成软标签（比 one-hot 更稳，也告诉网络哪些是次优选择）。
* 输出严格对齐 kimi3：棋盘 1/-1、encode() 三通道、pi 长度 size*size。
* 只用 numpy（不 import torch），torch 相关的东西延迟到 NetAgent 里再导入。
"""
from __future__ import annotations

import numpy as np

from engine import Engine
from game import GomokuGame


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def to_cells(game):
    """kimi3 的 board(1/-1) -> 引擎的 cells(1/2)。"""
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


def mask_legal(pi, legal):
    """把 pi 限制在合法着法上并归一化（兜底：全零时退回均匀分布）。"""
    pi = np.asarray(pi, dtype=np.float64).copy()
    keep = np.zeros(len(pi), dtype=bool)
    keep[legal] = True
    pi[~keep] = 0.0
    s = pi.sum()
    if s <= 0:
        pi[:] = 0.0
        pi[legal] = 1.0 / len(legal)
    else:
        pi /= s
    return pi.astype(np.float32)


# --------------------------------------------------------------------------- #
# 老师：传统引擎
# --------------------------------------------------------------------------- #
class TeacherAgent:
    """用路线 1 引擎出着 / 给软标签。"""

    def __init__(self, time_limit=0.2, max_depth=8, forbidden=True,
                 temperature=0.3, topk=4, vcf_depth=6):
        self.engine = Engine(time_limit=time_limit, max_depth=max_depth,
                             forbidden=forbidden, vcf_depth=vcf_depth)
        self.temperature = temperature
        self.topk = topk
        self.forbidden = forbidden
        self.stats = {"moves": 0, "time": 0.0, "vcf": 0}

    def policy(self, game):
        """返回长度 size*size 的策略分布（已按网页规则过滤禁手）。"""
        n = game.size
        self.engine.set_position(n, to_cells(game), 1 if game.current_player == 1 else 2)
        mv, info = self.engine.choose_move()
        legal = game.legal_moves()
        if mv is None or mv not in legal:
            # 极端情况：引擎没有着法，退回网页给出的合法点
            pi = np.zeros(n * n, dtype=np.float32)
            pi[legal] = 1.0 / len(legal)
            return pi
        self.stats["moves"] += 1
        self.stats["time"] += info.get("time", 0.0)
        if "VCF" in str(info.get("reason", "")):
            self.stats["vcf"] += 1
        soft = self.engine.root_policy(self.temperature, self.topk)
        pi = np.zeros(n * n, dtype=np.float64)
        for i, v in enumerate(soft):
            pi[i] = v
        if pi[mv] <= 0:                      # 强制着法 / 成五 / 算杀时没有 root_scores
            pi[:] = 0.0
            pi[mv] = 1.0
        return mask_legal(pi, legal)


# --------------------------------------------------------------------------- #
# 学生：kimi3 的策略价值网络 + MCTS
# --------------------------------------------------------------------------- #
class NetAgent:
    """强化模型 + 可选的"战术兜底"。

    战术兜底：每一步先用传统引擎快速看一眼，如果它判定为强制着法
    （能成五 / 必须挡四 / 算出 VCF 杀），就直接照走，不让 MCTS 去猜；
    其余局面仍然完全由神经网络 + 搜索决定。这样做是因为 90 次模拟在
    19×19 上只能看 3~4 层，而强制序列要 6~10 层。
    """
    TACTICAL_REASONS = {"成五", "挡四", "多四取优", "VCF 算杀"}

    def __init__(self, net, sims=100, temperature=0.0, add_noise=False,
                 tactical=False, tactical_time=0.15, forbidden=True,
                 guard_three=False):
        from mcts import MCTS                # 延迟导入，避免无 torch 环境炸掉
        self.mcts = MCTS(net)
        self.sims = sims
        self.temperature = temperature
        self.add_noise = add_noise
        self.tactical = tactical
        self.forbidden = forbidden
        self.guard_three = guard_three          # 对手有活三时，只允许在挡点里选
        self.engine = None
        if tactical:
            from engine import Engine
            self.engine = Engine(time_limit=tactical_time, forbidden=forbidden)

    def policy(self, game):
        if self.engine is not None:
            n = game.size
            cells = [int(v) for v in game.board.ravel()]          # 0 空 / 1 黑 / -1 白
            cells = [0 if v == 0 else (1 if v > 0 else 2) for v in cells]
            self.engine.forbidden = self.forbidden
            self.engine.set_position(n, cells, 1 if game.current_player == 1 else 2)
            mv, info = self.engine.choose_move()
            if mv is not None and info.get("reason") in self.TACTICAL_REASONS:
                pi = np.zeros(n * n, dtype=np.float64)
                pi[mv] = 1.0
                return pi
        pi = self.mcts.get_policy(game, self.sims, self.temperature, self.add_noise)
        if self.guard_three:
            from tactics import live_three_blocks, four_points, mask_policy
            found, blocks = live_three_blocks(game.board, game.current_player, game.size)
            if found and blocks:
                # 挡点的同时保留自己的反击手段（能成四的点），
                # 否则等于强迫模型放弃先手
                allowed = set(blocks) | four_points(game.board, game.current_player, game.size)
                pi = mask_policy(pi, allowed, game.size)
        return pi


# --------------------------------------------------------------------------- #
# 对局
# --------------------------------------------------------------------------- #
def play_game(policies, size, win_len=5, forbidden=True, sample=False, rng=None,
              opening_moves=0):
    """policies: {1: callable, -1: callable}，各自返回 pi。

    返回 (samples, winner, moves)。samples 里是 (encode, pi, z)。

    opening_moves > 0 时，先随机下若干手（在中盘附近均匀取点），用来打散开局。
    否则两个确定性 agent 会把同一盘棋重复下 N 遍，bench 的"6 局"其实只是 2 局。
    """
    rng = rng or np.random.default_rng()
    game = GomokuGame(size, win_len, use_forbidden=forbidden)
    records = []
    winner = 0
    if opening_moves > 0:
        mid = size // 2
        box = [r * size + c for r in range(max(0, mid - 4), min(size, mid + 5))
               for c in range(max(0, mid - 4), min(size, mid + 5))]
        for _ in range(opening_moves):
            legal = game.legal_moves()
            if not legal:
                break
            cand = [m for m in box if game.board.reshape(-1)[m] == 0]
            move = int(rng.choice(cand if cand else legal))
            finished, winner = game.step(move)
            if finished:
                break
    while winner == 0:
        legal = game.legal_moves()
        if not legal:
            break
        pi = policies[game.current_player](game)
        pi = mask_legal(pi, legal)
        records.append((game.encode(), pi, game.current_player))
        if sample:
            move = int(rng.choice(size * size, p=pi))
        else:
            move = int(np.argmax(pi))
        finished, winner = game.step(move)
        if finished:
            break
        if len(records) > size * size:
            break
    out = []
    for enc, pi, pl in records:
        z = 0.0 if winner == 0 else (1.0 if pl == winner else -1.0)
        out.append((enc, pi, z))
    return out, winner, len(records)


def augment(samples, size, sym=True):
    """8 种旋转/翻转做数据增强（复用 kimi3 的 symmetries）。"""
    if not sym:
        return [(s.astype(np.float32), p.astype(np.float32), z) for s, p, z in samples]
    helper = GomokuGame(size)
    out = []
    for s, p, z in samples:
        for s2, p2 in helper.symmetries(s, p):
            out.append((s2.astype(np.float32), p2.astype(np.float32), z))
    return out


def teacher_vs_teacher(games=100, size=9, win_len=5, forbidden=True,
                       time_limit=0.2, temperature=0.3, topk=4, seed=0,
                       sample_moves=True, sym_aug=True, verbose=True):
    """老师自己跟自己下，生成示范数据。"""
    rng = np.random.default_rng(seed)
    teacher = TeacherAgent(time_limit=time_limit, forbidden=forbidden,
                           temperature=temperature, topk=topk)
    data = []
    for g in range(games):
        samples, winner, nmoves = play_game(
            {1: teacher.policy, -1: teacher.policy},
            size, win_len, forbidden,
            sample=sample_moves, rng=rng)
        data.extend(augment(samples, size, sym_aug))
        if verbose and (g + 1) % 10 == 0:
            print(f"  老师对局 {g + 1}/{games}  最近一局 {nmoves} 手, "
                  f"胜方 {'黑' if winner == 1 else ('白' if winner == -1 else '和')}",
                  flush=True)
    if verbose:
        st = teacher.stats
        avg = st["time"] / max(1, st["moves"])
        print(f"  老师共 {st['moves']} 手，平均 {avg * 1000:.0f} ms/手，"
              f"算杀命中 {st['vcf']} 次", flush=True)
    return data


def net_vs_teacher(games=20, size=9, win_len=5, forbidden=True,
                   teacher=None, net_agent=None, seed=0, sym_aug=True, verbose=True):
    """网络 vs 老师：网络的着法用自己 MCTS 的策略，老师的着法用老师的软策略，
    终局结果 z 对双方都成立 —— 这就是“老师陪练”的专家迭代数据。"""
    if teacher is None or net_agent is None:
        raise ValueError("需要同时提供 teacher 和 net_agent")
    rng = np.random.default_rng(seed)
    data = []
    w = l = d = 0
    for g in range(games):
        net_black = (g % 2 == 0)
        policies = {1: net_agent.policy if net_black else teacher.policy,
                    -1: teacher.policy if net_black else net_agent.policy}
        samples, winner, nmoves = play_game(policies, size, win_len, forbidden)
        data.extend(augment(samples, size, sym_aug))
        if winner == 0:
            d += 1
            res = "和"
        elif (winner == 1) == net_black:
            w += 1
            res = "网络胜"
        else:
            l += 1
            res = "老师胜"
        if verbose:
            print(f"  陪练 {g + 1}/{games}: 网络执{'黑' if net_black else '白'}, "
                  f"{nmoves:3d} 手, {res}", flush=True)
    if verbose:
        print(f"  陪练合计: 网络 {w} 胜 / 老师 {l} 胜 / 和 {d}", flush=True)
    return data, {"net": w, "teacher": l, "draw": d}


def match(policy_a, policy_b, games, size, win_len=5, forbidden=True,
          swap=True, verbose=True, tag_a="A", tag_b="B", opening_moves=0, seed=0):
    """policy_a 与 policy_b 打 match（默认交替先后手）。返回 (a胜, b胜, 和)。

    opening_moves > 0：每局先随机下几手，保证"N 局"是 N 盘不同的棋，
    否则两个确定性 agent 会把同一盘重复下 N 遍（6 局其实只有 2 盘）。
    """
    wins_a = wins_b = draws = 0
    rng = np.random.default_rng(seed)
    for g in range(games):
        a_black = (g % 2 == 0) if swap else True
        pa = policy_a if a_black else policy_b
        pb = policy_b if a_black else policy_a
        _, winner, nmoves = play_game({1: pa, -1: pb}, size, win_len, forbidden,
                                      rng=rng, opening_moves=opening_moves)
        if winner == 0:
            draws += 1
            res = "和"
        elif (winner == 1) == a_black:
            wins_a += 1
            res = tag_a + "胜"
        else:
            wins_b += 1
            res = tag_b + "胜"
        if verbose:
            side = "黑" if a_black else "白"
            print(f"  第 {g + 1:2d} 局: {tag_a}执{side}, {nmoves:3d} 手, {res}", flush=True)
    return wins_a, wins_b, draws


def score_to_elo(s):
    """胜率 -> 分差 (Elo)。s 在 (0,1) 内。"""
    s = min(max(s, 1e-3), 1 - 1e-3)
    return -400.0 * np.log10(1.0 / s - 1.0)

