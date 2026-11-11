"""五子棋「路线 1」引擎：棋型评估 + α-β 搜索 + 强制着法 + VCF 算杀。

纯 Python，不依赖 numpy / torch，CPU 上单步通常 0.1~2 秒。

两个用途：
  1. server_ab.py 的 AI 后端 —— 与 kimi3 的网页版共用同一套 HTTP 协议；
  2. teacher.py 的“老师” —— 给 AlphaZero 网络做监督预训练 / 软标签（见 pretrain.py）。

算法结构
--------
1. 棋盘：一维列表。4 个方向的全部“长度 5 窗口”共 n_win 个，每个窗口维护一个
   三进制编码（0 空 / 1 黑 / 2 白）。落子只影响 20 个窗口 → 评估完全增量，O(1)。
2. 评估：窗口编码 → 分值查表（成五 / 四 / 活三 / 二 ...），双方分差，防守略加权。
3. 威胁表：增量维护“某方下一手就能成五”的点（纯四窗口里的空位）。搜索每个节点
   先查威胁表：自己有四点 → 立刻赢；对方有四点 → 只能挡，多于一个 → 必败。
   这一步让引擎在浅层搜索下也不会漏杀、漏挡。
4. 候选点打分缓存：每个空点缓存“黑下这里得分/白下这里得分”，只有窗口受影响的
   格点会被标脏，取候选时再算。这样每个节点都能扫描全盘的邻近空点（不会漏掉
   远处的活三），代价仍然可控。
5. 搜索：negamax + α-β + 置换表（Zobrist）+ 迭代加深；候选点按“进攻分 + 防守分”
   取前若干个，越靠近根节点取得越多。
6. 算杀：VCF（连续冲四），找 5~10 步内的强制胜。
7. 禁手：长连 / 四四 / 三三。根节点用严格版（能识别跳三、跳四，比网页版更严），
   搜索内部用与网页一致的连续型快速判定，保证最终走出来的着法一定合规。
"""

from __future__ import annotations

import heapq
import math
import time

EMPTY, BLACK, WHITE = 0, 1, 2
WIN = 10 ** 7
INF = 10 ** 9

_POW3 = (1, 3, 9, 27, 81)
_PURE = (0, 2, 20, 200, 2000)      # 纯己方子数 -> 窗口分值

# 3^5 = 243 个窗口编码的查表
_TB = [0] * 243                    # 黑棋视角的分值
_TW = [0] * 243                    # 白棋视角的分值
_FOUR_MASK = [0] * 243             # 1=黑冲四窗口, 2=白冲四窗口, 0=不是
_EMPTY_IDX = [-1] * 243            # 唯一空位的下标（用于找“成五点”）


def _build_tables():
    for code in range(243):
        d = [(code // _POW3[i]) % 3 for i in range(5)]
        b = d.count(1)
        w = d.count(2)
        if b == 5:
            _TB[code] = WIN
        elif w == 5:
            _TW[code] = WIN
        else:
            if w == 0:
                _TB[code] = _PURE[b]
            if b == 0:
                _TW[code] = _PURE[w]
        if b == 4 and w == 0:
            _FOUR_MASK[code] = 1
        elif w == 4 and b == 0:
            _FOUR_MASK[code] = 2
        if d.count(0) == 1:
            _EMPTY_IDX[code] = d.index(0)


_build_tables()

_DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


def _max_run(seg):
    best = run = 0
    for v in seg:
        if v == 1:
            run += 1
            if run > best:
                best = run
        else:
            run = 0
    return best


def _has_open_four(seg):
    """seg 里是否存在“两端皆空”的四连（活四）。"""
    padded = [2] + list(seg) + [2]
    for i in range(1, 7):
        if (padded[i] == 1 and padded[i + 1] == 1 and padded[i + 2] == 1
                and padded[i + 3] == 1 and padded[i - 1] == 0 and padded[i + 4] == 0):
            return True
    return False


class _Timeout(Exception):
    pass


# --------------------------------------------------------------------------- #
# 棋盘几何：窗口 / 邻域 / 脏格关联表，按棋盘大小缓存
# --------------------------------------------------------------------------- #
class Geometry:
    __slots__ = ("size", "n", "cell_wins", "win_cells", "neighbors", "nwin",
                 "cell_watch")

    def __init__(self, size):
        n = size
        self.size = size
        self.n = n * n
        cell_wins = [[] for _ in range(n * n)]
        win_cells = []

        for dr, dc in _DIRS:
            for r in range(n):
                for c in range(n):
                    pr, pc = r - dr, c - dc
                    if 0 <= pr < n and 0 <= pc < n:
                        continue                      # 不是这条线的起点
                    line = []
                    rr, cc = r, c
                    while 0 <= rr < n and 0 <= cc < n:
                        line.append(rr * n + cc)
                        rr += dr
                        cc += dc
                    for s in range(len(line) - 4):
                        wid = len(win_cells)
                        cells = tuple(line[s:s + 5])
                        win_cells.append(cells)
                        for pos, cid in enumerate(cells):
                            cell_wins[cid].append((wid, _POW3[pos]))

        self.cell_wins = cell_wins
        self.win_cells = win_cells
        self.nwin = len(win_cells)

        neighbors = []
        for r in range(n):
            for c in range(n):
                lst = []
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        if dr or dc:
                            rr, cc = r + dr, c + dc
                            if 0 <= rr < n and 0 <= cc < n:
                                lst.append(rr * n + cc)
                neighbors.append(lst)
        self.neighbors = neighbors

        # cell_watch[c]: 与 c 共享任一窗口的格点（含自己），落子后这些点的分数要重算
        watch = []
        for c in range(n * n):
            seen = {c}
            for wid, _mult in cell_wins[c]:
                seen.update(win_cells[wid])
            watch.append(tuple(seen))
        self.cell_watch = watch


_GEO = {}


def geometry(size):
    g = _GEO.get(size)
    if g is None:
        g = _GEO[size] = Geometry(size)
    return g


_ZOB = {}


def _zobrist(size):
    z = _ZOB.get(size)
    if z is None:
        rnd = 0x9E3779B97F4A7C15
        tbl = [[0] * (size * size) for _ in range(3)]
        for p in (1, 2):
            for c in range(size * size):
                rnd = (rnd * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
                tbl[p][c] = rnd
        z = _ZOB[size] = tbl
    return z


# --------------------------------------------------------------------------- #
# 引擎
# --------------------------------------------------------------------------- #
class Engine:
    """传统搜索引擎。一局之内可以复用同一个对象（置换表跨步有效）。"""

    def __init__(self, time_limit=1.2, max_depth=8, forbidden=True, vcf_depth=6):
        self.time_limit = float(time_limit)
        self.max_depth = int(max_depth)
        self.forbidden = bool(forbidden)
        self.vcf_depth = int(vcf_depth)

        self.geo = None
        self.size = 0
        self.n = 0
        self.cells = []
        self.code = []
        self.score = [0, 0, 0]
        self.near = []
        self.tcnt = [None, {}, {}]
        self.hash = 0
        self.turn = BLACK
        self.last = -1

        self.gB = []          # 黑在某空点落子能拿到的分（缓存）
        self.gW = []          # 白在某空点落子能拿到的分（缓存）
        self.dirty = []
        self.dirty_list = []

        self.tt = {}
        self.tt_cap = 300000
        # 消融开关：默认全开，行为与原来完全一致；只用于 tools/ablation_engine.py 的对照实验
        self.use_tt = True          # 置换表
        self.use_threat = True      # 威胁表（成五/挡四的强制着法）
        self.use_vcf = True         # VCF 算杀
        self.deadline = 0.0
        self.nodes = 0
        self.root_scores = {}

    # ---------------- 局面装载 ---------------- #
    def set_position(self, size, cells, player):
        """cells: 长度 size*size 的一维列表, 0 空 / 1 黑 / 2 白; player 是轮到谁。"""
        geo = geometry(size)
        if self.size != size:
            self.size = size
            self.n = size * size
            self.geo = geo
            self.zob = _zobrist(size)
            self.tt.clear()
            self.gB = [0] * self.n
            self.gW = [0] * self.n
            self.dirty = [True] * self.n
            self.dirty_list = list(range(self.n))
            self.near = [0] * self.n
        self.cells = list(cells)
        if len(self.cells) != self.n:
            raise ValueError(f"棋盘大小不符: 期望 {self.n}, 收到 {len(self.cells)}")
        self.code = [0] * geo.nwin
        self.score = [0, 0, 0]
        self.near = [0] * self.n
        self.tcnt = [None, {}, {}]
        self.hash = 0
        self.turn = player
        self.last = -1
        self.dirty = [True] * self.n
        self.dirty_list = list(range(self.n))
        for c, v in enumerate(self.cells):
            if v:
                self._place_raw(c, v)

    # ---------------- 增量维护 ---------------- #
    def _mark_dirty(self, c):
        if not self.dirty[c]:
            self.dirty[c] = True
            self.dirty_list.append(c)

    def _place_raw(self, c, p):
        self.cells[c] = p
        self.hash ^= self.zob[p][c]
        code = self.code
        score = self.score
        fm = _FOUR_MASK
        for wid, mult in self.geo.cell_wins[c]:
            old = code[wid]
            new = old + p * mult
            code[wid] = new
            score[BLACK] += _TB[new] - _TB[old]
            score[WHITE] += _TW[new] - _TW[old]
            if fm[old]:
                self._threat_dec(old, wid)
            if fm[new]:
                self._threat_inc(new, wid)
        for nb in self.geo.neighbors[c]:
            self.near[nb] += 1
        dirty = self.dirty
        for x in self.geo.cell_watch[c]:
            if not dirty[x]:
                dirty[x] = True
                self.dirty_list.append(x)

    def _unplace_raw(self, c, p):
        code = self.code
        score = self.score
        fm = _FOUR_MASK
        for wid, mult in self.geo.cell_wins[c]:
            old = code[wid]
            new = old - p * mult
            code[wid] = new
            score[BLACK] -= _TB[old] - _TB[new]
            score[WHITE] -= _TW[old] - _TW[new]
            if fm[old]:
                self._threat_dec(old, wid)
            if fm[new]:
                self._threat_inc(new, wid)
        self.cells[c] = EMPTY
        self.hash ^= self.zob[p][c]
        for nb in self.geo.neighbors[c]:
            self.near[nb] -= 1
        dirty = self.dirty
        for x in self.geo.cell_watch[c]:
            if not dirty[x]:
                dirty[x] = True
                self.dirty_list.append(x)

    def _threat_inc(self, code, wid):
        p = _FOUR_MASK[code]
        cell = self.geo.win_cells[wid][_EMPTY_IDX[code]]
        d = self.tcnt[p]
        d[cell] = d.get(cell, 0) + 1

    def _threat_dec(self, code, wid):
        p = _FOUR_MASK[code]
        cell = self.geo.win_cells[wid][_EMPTY_IDX[code]]
        d = self.tcnt[p]
        k = d.get(cell, 0) - 1
        if k > 0:
            d[cell] = k
        else:
            d.pop(cell, None)

    def _refresh_dirty(self):
        """把标脏的格点的“落子收益”重算一遍（只影响候选排序，不影响正确性）。"""
        if not self.dirty_list:
            return
        cells = self.cells
        code = self.code
        cw = self.geo.cell_wins
        gB = self.gB
        gW = self.gW
        dirty = self.dirty
        for c in self.dirty_list:
            if not dirty[c]:
                continue
            dirty[c] = False
            if cells[c]:
                continue
            gb = gw = 0
            for wid, mult in cw[c]:
                old = code[wid]
                gb += _TB[old + mult] - _TB[old]
                gw += _TW[old + 2 * mult] - _TW[old]
            gB[c] = gb
            gW[c] = gw
        self.dirty_list.clear()

    def _make(self, c, p):
        self._place_raw(c, p)
        self.last = c
        self.turn = 3 - p

    def _unmake(self, c, p):
        self._unplace_raw(c, p)
        self.turn = p

    # ---------------- 评估 ---------------- #
    def _gain2(self, c, me):
        """直接算：我在 c 落子的收益 + 对手在 c 落子的收益（不依赖缓存）。"""
        code = self.code
        opp = 3 - me
        T1 = _TB if me == BLACK else _TW
        T2 = _TW if me == BLACK else _TB
        g1 = g2 = 0
        for wid, mult in self.geo.cell_wins[c]:
            old = code[wid]
            g1 += T1[old + me * mult] - T1[old]
            g2 += T2[old + opp * mult] - T2[old]
        return g1, g2

    def _evaluate(self, me):
        b = self.score[BLACK]
        w = self.score[WHITE]
        if me == BLACK:
            return b - (w + (w >> 3) + (w >> 4))
        return w - (b + (b >> 3) + (b >> 4))

    # ---------------- 禁手 ---------------- #
    def _line9(self, move, dr, dc):
        n = self.size
        r, c = divmod(move, n)
        cells = self.cells
        seg = []
        for k in range(-4, 5):
            rr, cc = r + k * dr, c + k * dc
            if 0 <= rr < n and 0 <= cc < n:
                seg.append(cells[rr * n + cc])
            else:
                seg.append(2)                    # 墙视作对方子
        return seg

    def _forbidden_strict(self, move):
        """严格禁手判定（仅黑棋）：长连 / 四四 / 三三，含跳三、跳四。"""
        cells = self.cells
        if cells[move] != EMPTY:
            return False
        cells[move] = BLACK
        try:
            fours = threes = 0
            for dr, dc in _DIRS:
                seg = self._line9(move, dr, dc)
                if _max_run(seg) >= 6:
                    return True                  # 长连
                five = three = False
                for i in range(9):
                    if seg[i] != EMPTY:
                        continue
                    seg[i] = 1
                    run = _max_run(seg)
                    if run >= 5:
                        five = True
                    elif run == 4 and _has_open_four(seg):
                        three = True
                    seg[i] = EMPTY
                if five:
                    fours += 1
                elif three:
                    threes += 1
                if fours >= 2 or threes >= 2:
                    return True
            return False
        finally:
            cells[move] = EMPTY

    def _forbidden_quick(self, move):
        """快速禁手判定（与网页 gomoku.html 的规则一致：只认连续棋形）。"""
        n = self.size
        r, c = divmod(move, n)
        cells = self.cells
        if cells[move] != EMPTY:
            return False
        cells[move] = BLACK
        try:
            fours = threes = 0
            for dr, dc in _DIRS:
                run = 1
                opens = 0
                for s in (1, -1):
                    rr, cc = r + s * dr, c + s * dc
                    while 0 <= rr < n and 0 <= cc < n and cells[rr * n + cc] == BLACK:
                        run += 1
                        rr += s * dr
                        cc += s * dc
                    if 0 <= rr < n and 0 <= cc < n and cells[rr * n + cc] == EMPTY:
                        opens += 1
                if run == 5:
                    return False                 # 成五优先，不算禁手
                if run > 5:
                    return True                  # 长连
                if run == 4 and opens >= 1:
                    fours += 1
                elif run == 3 and opens == 2:
                    threes += 1
                if fours >= 2 or threes >= 2:
                    return True
            return False
        finally:
            cells[move] = EMPTY

    def is_legal(self, move, me):
        if self.cells[move] != EMPTY:
            return False
        if me != BLACK or not self.forbidden:
            return True
        return not self._forbidden_strict(move)

    def _legal_quick(self, move, me):
        if self.cells[move] != EMPTY:
            return False
        if me != BLACK or not self.forbidden:
            return True
        return not self._forbidden_quick(move)

    # ---------------- 候选生成 ---------------- #
    def _gen(self, me, limit, tt_move=-1):
        """全盘邻近空点按“进攻 + 0.9*防守”排序，取前 limit 个。"""
        self._refresh_dirty()
        cells = self.cells
        near = self.near
        if me == BLACK:
            G1 = self.gB
            G2 = self.gW
        else:
            G1 = self.gW
            G2 = self.gB
        tmp = []
        for c in range(self.n):
            if cells[c] == EMPTY and near[c] > 0:
                tmp.append((G1[c] + (G2[c] * 9) // 10, c))
        if not tmp:                              # 空盘/孤子：随便给中心附近
            mid = self.size // 2
            out = [mid * self.size + mid]
            for c in range(self.n):
                if len(out) >= limit:
                    break
                if cells[c] == EMPTY:
                    out.append(c)
            return out
        if limit < len(tmp):
            out = [c for _, c in heapq.nlargest(limit, tmp)]
        else:
            tmp.sort(reverse=True)
            out = [c for _, c in tmp]
        if tt_move >= 0 and tt_move not in out and cells[tt_move] == EMPTY:
            out.insert(0, tt_move)
            del out[limit:]
        return out

    def _limit(self, ply):
        if ply <= 0:
            return 20
        if ply == 1:
            return 12
        if ply == 2:
            return 10
        return 8

    # ---------------- 搜索 ---------------- #
    def negamax(self, depth, alpha, beta, ply):
        self.nodes += 1
        if not (self.nodes & 255) and time.perf_counter() > self.deadline:
            raise _Timeout

        me = self.turn
        opp = 3 - me

        # 1) 自己有冲四点 -> 下一手成五，必胜
        if self.use_threat and self.tcnt[me]:
            return WIN - ply

        # 2) 置换表
        key = self.hash
        ent = self.tt.get(key) if self.use_tt else None
        tt_move = -1
        if ent is not None:
            d, flag, val, tt_move = ent
            if d >= depth:
                if flag == 0:
                    return val
                if flag == 1 and val >= beta:
                    return val
                if flag == 2 and val <= alpha:
                    return val

        # 3) 叶子
        if depth <= 0:
            if self.tcnt[opp] and ply < 40:
                depth = 1                        # 强制着法延伸（伪静息搜索）
            else:
                return self._evaluate(me)

        # 4) 对方有冲四点 -> 只能挡
        if self.use_threat and self.tcnt[opp]:
            pts = [c for c in self.tcnt[opp] if self._legal_quick(c, me)]
            if not pts:
                return -(WIN - ply)              # 挡不了，必输
            if len(pts) > 1:
                return -(WIN - ply - 1)          # 双四，挡不住
            m = pts[0]
            prev = self.last
            self._make(m, me)
            try:
                val = -self.negamax(depth - 1, -beta, -alpha, ply + 1)
            finally:
                self._unmake(m, me)
                self.last = prev
            return val

        ordered = self._gen(me, self._limit(ply), tt_move)
        if not ordered:
            return 0

        best_val = -INF
        best_move = ordered[0]
        alpha0 = alpha
        a = alpha
        for m in ordered:
            if not self._legal_quick(m, me):
                continue
            prev = self.last
            self._make(m, me)
            try:
                val = -self.negamax(depth - 1, -beta, -a, ply + 1)
            finally:
                self._unmake(m, me)
                self.last = prev
            if val > best_val:
                best_val = val
                best_move = m
                if val > a:
                    a = val
                    if a >= beta:
                        break

        if best_val == -INF:
            return self._evaluate(me)

        if best_val <= alpha0:
            flag = 2
        elif best_val >= beta:
            flag = 1
        else:
            flag = 0
        if self.use_tt:
            if len(self.tt) > self.tt_cap:
                self.tt.clear()
            self.tt[key] = (depth, flag, best_val, best_move)
        return best_val

    # ---------------- VCF 算杀 ---------------- #
    def _four_moves(self, me):
        self._refresh_dirty()
        cells = self.cells
        near = self.near
        G = self.gB if me == BLACK else self.gW
        out = []
        for c in range(self.n):
            if cells[c] == EMPTY and near[c] > 0 and G[c] >= 1500:
                out.append(c)
        out.sort(key=lambda c: -G[c])
        return out

    def _vcf(self, me, depth, ply):
        """连续冲四搜索：返回一个能强制取胜的着法，或 None。"""
        if self.tcnt[me]:
            pts = [c for c in self.tcnt[me] if self._legal_quick(c, me)]
            if pts:
                return pts[0]
        if depth <= 0 or ply > 24:
            return None
        if time.perf_counter() > self.deadline:
            raise _Timeout

        opp = 3 - me
        for m in self._four_moves(me):
            if not self._legal_quick(m, me):
                continue
            prev = self.last
            self._make(m, me)
            try:
                my_pts = [c for c in self.tcnt[me] if self._legal_quick(c, me)]
                if not my_pts:
                    continue
                if self.tcnt[opp]:
                    continue                     # 对方能先成五，冲四无效
                if len(my_pts) >= 2:
                    return m                     # 双四 / 活四，对方挡不住
                block = my_pts[0]
                if opp == BLACK and self.forbidden and self._forbidden_quick(block):
                    return m                     # 对方被禁手挡住，无法防守
                self._make(block, opp)
                try:
                    if self._vcf(me, depth - 1, ply + 2) is not None:
                        return m
                finally:
                    self._unmake(block, opp)
                    self.last = m
            finally:
                self._unmake(m, me)
                self.last = prev
        return None

    # ---------------- 根节点 ---------------- #
    def _root_search(self, depth, cands):
        me = self.turn
        alpha = -INF
        beta = INF
        best_val = -INF
        best_move = None
        scores = {}
        for m in cands:
            if not self.is_legal(m, me):
                continue
            prev = self.last
            self._make(m, me)
            try:
                val = -self.negamax(depth - 1, -beta, -alpha, 1)
            finally:
                self._unmake(m, me)
                self.last = prev
            scores[m] = val
            if val > best_val:
                best_val = val
                best_move = m
                if val > alpha:
                    alpha = val
        # 只有完整跑完这一层才覆盖，避免超时把上一层的分值清空
        self.root_scores = scores
        return best_val, best_move

    def choose_move(self):
        """为 self.turn 选一个着法。返回 (move, info)。"""
        t0 = time.perf_counter()
        self.nodes = 0
        self.root_scores = {}
        self.deadline = t0 + self.time_limit
        me = self.turn
        opp = 3 - me
        info = {"reason": "", "depth": 0, "score": 0, "nodes": 0, "time": 0.0}

        if not any(self.cells):
            c = (self.size // 2) * self.size + self.size // 2
            info.update(reason="开局天元")
            info["time"] = time.perf_counter() - t0
            return c, info

        # 1) 自己能成五
        if self.use_threat and self.tcnt[me]:
            cands = [c for c in self.tcnt[me] if self.is_legal(c, me)]
            if cands:
                info.update(reason="成五", score=WIN)
                info["time"] = time.perf_counter() - t0
                return cands[0], info

        # 2) 对方有四点，必须挡
        if self.use_threat and self.tcnt[opp]:
            blocks = [c for c in self.tcnt[opp] if self.is_legal(c, me)]
            if blocks:
                if len(blocks) == 1:
                    best = blocks[0]
                else:
                    def _key(c):
                        g1, g2 = self._gain2(c, me)
                        return g1 + g2
                    best = max(blocks, key=_key)
                info.update(reason="挡四" if len(blocks) == 1 else "多四取优",
                            score=-(WIN - 1))
                info["time"] = time.perf_counter() - t0
                return best, info

        # 3) VCF 算杀
        vcf = None
        if self.use_vcf and self.use_threat:
            try:
                vcf = self._vcf(me, self.vcf_depth, 0)
            except _Timeout:
                vcf = None
        if vcf is not None and self.is_legal(vcf, me):
            info.update(reason="VCF 算杀", score=WIN - 1)
            info["time"] = time.perf_counter() - t0
            return vcf, info

        # 4) 迭代加深的 α-β
        cands = self._gen(me, 24)
        if not cands:
            info["time"] = time.perf_counter() - t0
            return None, info

        best = cands[0]
        done = 0
        try:
            for depth in range(2, self.max_depth + 1):
                val, mv = self._root_search(depth, cands)
                if mv is None:
                    break
                best = mv
                done = depth
                info["score"] = val
                if abs(val) > WIN // 2:
                    break
                cands.remove(mv)
                cands.insert(0, mv)
        except _Timeout:
            pass

        info.update(reason="α-β", depth=done)
        info["nodes"] = self.nodes
        info["time"] = time.perf_counter() - t0
        return best, info

    # ---------------- 软标签（给蒸馏用） ---------------- #
    def root_policy(self, temperature=0.5, topk=6):
        """把根节点各着法的分值转成概率分布（长度 n），用于给网络做软标签。"""
        pi = [0.0] * self.n
        if not self.root_scores:
            return pi
        items = sorted(self.root_scores.items(), key=lambda kv: -kv[1])[:topk]
        best = items[0][1]
        scale = max(1.0, 400.0 * max(float(temperature), 1e-3))
        weights = []
        for m, v in items:
            w = math.exp(max(-30.0, (v - best) / scale))
            weights.append((m, w))
        total = sum(w for _, w in weights)
        if total <= 0:
            pi[items[0][0]] = 1.0
            return pi
        for m, w in weights:
            pi[m] = w / total
        return pi
