"""蒙特卡洛树搜索 (PUCT)，AlphaZero 风格。

约定: 每个 Node 记录的 q 值都是"从该节点轮到谁走棋"那一方的视角。
"""
import math
import numpy as np


class Node:
    __slots__ = ("prior", "player", "children", "visits", "value_sum")

    def __init__(self, prior, player):
        self.prior = prior          # 父节点指向本节点的先验概率
        self.player = player        # 本节点轮到谁走
        self.children = {}          # move -> Node
        self.visits = 0
        self.value_sum = 0.0

    @property
    def q(self):
        return self.value_sum / self.visits if self.visits else 0.0


class MCTS:
    def __init__(self, net, c_puct=1.5, dirichlet_alpha=0.3, noise_eps=0.25):
        self.net = net
        self.c_puct = c_puct
        self.alpha = dirichlet_alpha
        self.noise_eps = noise_eps

    def _expand(self, node, game, add_noise=False):
        """用神经网络扩展叶子节点，返回该局面价值(从 node.player 视角)。"""
        probs, value = self.net.predict(game.encode())
        legal = game.legal_moves()
        if not legal:                       # 没有合法点（棋盘满/黑棋全是禁手点）：不扩展
            return value
        priors = np.zeros_like(probs)
        priors[legal] = probs[legal]
        total = priors.sum()
        if total > 0:
            priors = priors / total
        else:
            # 网络在合法点上的概率全是 0（极端但不该发生）：退回合法点上的均匀分布，
            # 绝不能除零、也不能把概率铺到已占点上
            priors[:] = 0.0
            priors[legal] = 1.0 / len(legal)
        if add_noise:
            noise = np.zeros_like(probs)
            noise[legal] = np.random.dirichlet([self.alpha] * len(legal))
            priors = (1 - self.noise_eps) * priors + self.noise_eps * noise
        for m in legal:
            node.children[m] = Node(float(priors[m]), -game.current_player)
        return value

    def search(self, game, num_sims, add_noise=False):
        root = Node(1.0, game.current_player)
        self._expand(root, game, add_noise=add_noise)
        for _ in range(num_sims):
            g = game.copy()
            node = root
            path = [node]
            # 选择: 从父节点玩家视角最大化 -child.q + U
            finished, winner = False, 0
            while node.children:
                sqrt_parent = math.sqrt(max(1, node.visits))
                best_score, best_move, best_child = -1e18, None, None
                for m, child in node.children.items():
                    u = self.c_puct * child.prior * sqrt_parent / (1 + child.visits)
                    score = -child.q + u
                    if score > best_score:
                        best_score, best_move, best_child = score, m, child
                finished, winner = g.step(best_move)
                node = best_child
                path.append(node)
                if finished:
                    break
            # 评估: 价值统一为叶子节点 node.player 的视角
            if finished:
                value = 0.0 if winner == 0 else -1.0  # 获胜者是刚走棋的一方
            else:
                value = self._expand(node, g)
            # 回传
            leaf_player = node.player
            for n in reversed(path):
                n.visits += 1
                n.value_sum += value if n.player == leaf_player else -value
        return root

    def get_policy(self, game, num_sims, temperature=1.0, add_noise=False):
        """返回访问计数归一化得到的策略向量 (长度 size*size)。"""
        root = self.search(game, num_sims, add_noise=add_noise)
        counts = np.zeros(game.size * game.size, dtype=np.float64)
        for m, child in root.children.items():
            counts[m] = child.visits
        if counts.sum() <= 0:
            # 一次访问都没有（极端情况）：退回合法点上的均匀分布，
            # 不能用 np.full_like（那会把概率铺到已占点上，让 argmax 选中非法点）
            legal = game.legal_moves()
            pi = np.zeros_like(counts)
            if legal:
                pi[legal] = 1.0 / len(legal)
            return pi
        if temperature < 1e-3:
            pi = np.zeros_like(counts)
            pi[int(counts.argmax())] = 1.0
        else:
            c = counts ** (1.0 / temperature)
            total = c.sum()
            pi = c / total if total > 0 else np.full_like(c, 1.0 / len(c))
        return pi
