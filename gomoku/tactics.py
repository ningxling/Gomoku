"""战术层：识别"活三"并给出该挡的点。

活三的判定（与规则无关，纯棋形）：
    对手在某个空点落子后，能形成"活四"（两个成五点）→ 该点就是活三的延伸点，
    当前局面就存在活三威胁。

该挡的点 = {那个延伸点本身} ∪ {落子后形成的两个成五点}

处理方式只做"候选剪枝"：把搜索限制在这些点上，具体挡哪一边仍由搜索决定。
不做成 one-hot 标签——活三不像"成五/挡四"那样是规则强制的。
"""
import numpy as np

DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


def five_points_after(b, p, r, c, n):
    """假设 p 在 (r,c) 落子后，p 能一手成五的空点（去重）。"""
    b[r, c] = p
    pts = set()
    for dr, dc in DIRS:
        for k in range(5):
            start = -(4 - k)
            cells = [(r + (start + j) * dr, c + (start + j) * dc) for j in range(5)]
            if any(x < 0 or x >= n or y < 0 or y >= n for x, y in cells):
                continue
            vals = [b[x, y] for x, y in cells]
            if vals.count(p) == 4 and vals.count(0) == 1:
                pts.add(cells[vals.index(0)])
    b[r, c] = 0
    return pts


def _extensions(b, opp, n):
    """对手所有"落这里能成活四"的点（以及各自的成五点）。"""
    ext = []
    for r in range(n):
        for c in range(n):
            if b[r, c] != 0:
                continue
            r0, r1 = max(0, r - 2), min(n, r + 3)
            c0, c1 = max(0, c - 2), min(n, c + 3)
            if not (b[r0:r1, c0:c1] == opp).any():
                continue
            pts = five_points_after(b, opp, r, c, n)
            if len(pts) >= 2:
                ext.append(((r, c), pts))
    return ext


def live_three_blocks(board, me, n=None):
    """返回 (是否面临活三, 该挡的点集合{一维下标})。

    挡点用"精确"判定：把候选点填上我的子之后，对手**再也做不出活四**才算有效挡点。
    这样不会把"挡了也没用"的点混进来。
    """
    b = np.asarray(board)
    n = n or b.shape[0]
    opp = -me
    ext = _extensions(b, opp, n)
    if not ext:
        return False, set()
    # 候选：每个活三的延伸点 + 它成活的四的两个端点
    cand = set()
    for (r, c), pts in ext:
        cand.add((r, c))
        cand |= pts
    blocks = set()
    for (r, c) in cand:
        if b[r, c] != 0:
            continue
        b[r, c] = me                      # 试着我先占上
        still = _extensions(b, opp, n)    # 对手还有没有活四手段
        b[r, c] = 0
        if not still:
            blocks.add(r * n + c)
    return True, blocks


def four_points(board, me, n=None):
    """我自己"落一子能成四"的点（用于抢 tempo 反攻）。"""
    b = np.asarray(board)
    n = n or b.shape[0]
    out = set()
    for r in range(n):
        for c in range(n):
            if b[r, c] != 0:
                continue
            if five_points_after(b, me, r, c, n):
                out.add(r * n + c)
    return out


def mask_policy(pi, allowed, n):
    """把策略限制在 allowed 上并归一化；allowed 为空时原样返回。"""
    if not allowed:
        return pi
    m = np.zeros_like(pi)
    for i in allowed:
        m[i] = pi[i]
    s = m.sum()
    if s <= 0:                       # 搜索没给这些点任何权重：均匀铺开
        for i in allowed:
            m[i] = 1.0
        return m
    return m / s
