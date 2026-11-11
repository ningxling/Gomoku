"""五子棋核心逻辑：棋盘、落子、胜负判定、神经网络输入编码。

棋盘用 numpy 数组表示: 0=空, 1=黑, -1=白。黑先走。
"""
import numpy as np


class GomokuGame:
    def __init__(self, size=9, win_len=5, use_forbidden=True):
        self.size = size
        self.win_len = win_len
        self.use_forbidden = use_forbidden
        self.board = np.zeros((size, size), dtype=np.int8)
        self.current_player = 1  # 黑先
        self.last_move = None
        self.n_stones = 0        # 已落子数，用于 O(1) 判断棋盘是否已满

    def copy(self):
        g = GomokuGame(self.size, self.win_len, self.use_forbidden)
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.last_move = self.last_move
        g.n_stones = self.n_stones
        return g

    def legal_moves(self):
        """返回可落子位置的一维索引列表 (0 .. size*size-1)，排除禁手点。"""
        moves = np.flatnonzero(self.board.ravel() == 0).tolist()
        if self.use_forbidden and self.current_player == 1:
            moves = [m for m in moves if not self.is_forbidden(m)]
        return moves

    def is_forbidden(self, move):
        """黑棋禁手判定: 长连 / 双四 / 双活三。五连优先(成五即胜不算禁手)，白棋无禁手。

        简化实现: 只识别连续型棋形, 不含跳三、跳四。
        """
        r, c = divmod(move, self.size)
        if self.board[r, c] != 0:
            return False
        b = self.board
        b[r, c] = 1
        try:
            five = overline = False
            fours = threes = 0
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                run = 1
                open_ends = 0
                for sign in (1, -1):
                    nr, nc = r + sign * dr, c + sign * dc
                    while 0 <= nr < self.size and 0 <= nc < self.size and b[nr, nc] == 1:
                        run += 1
                        nr += sign * dr
                        nc += sign * dc
                    if 0 <= nr < self.size and 0 <= nc < self.size and b[nr, nc] == 0:
                        open_ends += 1
                if run == self.win_len:
                    five = True
                elif run > self.win_len:
                    overline = True
                elif run == self.win_len - 1 and open_ends >= 1:
                    fours += 1
                elif run == self.win_len - 2 and open_ends == 2:
                    threes += 1
            if five:
                return False
            return overline or fours >= 2 or threes >= 2
        finally:
            b[r, c] = 0

    def step(self, move):
        """在 move(一维索引)处落子，切换玩家。返回 (finished, winner)。"""
        r, c = divmod(move, self.size)
        assert self.board[r, c] == 0, f"非法落子: ({r}, {c})"
        self.board[r, c] = self.current_player
        self.n_stones += 1
        self.last_move = move
        winner = self._check_win(r, c)
        # 用落子数判断"棋盘是否已满"，而不是调用 legal_moves()
        # （黑棋的 legal_moves 要对所有空点跑禁手判定，是搜索里最大的开销）
        finished = winner != 0 or self.n_stones >= self.size * self.size
        self.current_player *= -1
        return finished, winner

    def _check_win(self, r, c):
        p = self.board[r, c]
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1
            for sign in (1, -1):
                nr, nc = r + sign * dr, c + sign * dc
                while 0 <= nr < self.size and 0 <= nc < self.size and self.board[nr, nc] == p:
                    count += 1
                    nr += sign * dr
                    nc += sign * dc
            if count >= self.win_len:
                return int(p)
        return 0

    def encode(self):
        """从当前玩家视角编码局面: [当前玩家的子, 对方的子, 先手标识] 三个通道。"""
        p = self.current_player
        mine = (self.board == p).astype(np.float32)
        theirs = (self.board == -p).astype(np.float32)
        side = np.full((self.size, self.size), float(p == 1), dtype=np.float32)
        return np.stack([mine, theirs, side])

    def symmetries(self, board_encoded, pi):
        """棋盘 8 种旋转/翻转对称变换，用于数据增强。"""
        n = self.size
        pi_2d = pi.reshape(n, n)
        out = []
        for k in range(4):
            b = np.rot90(board_encoded, k, axes=(1, 2))
            p = np.rot90(pi_2d, k)
            out.append((b, p.ravel()))
            out.append((np.flip(b, axis=2), np.flip(p, axis=1).ravel()))
        return out
