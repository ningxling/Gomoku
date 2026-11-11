"""策略-价值网络：小型残差卷积网络 (全卷积, 权重与棋盘尺寸无关)。

输入: (batch, 3, size, size) — size 任意
输出: policy logits (batch, size*size), value (batch, 1) 范围 [-1, 1]

策略头用 1x1 卷积输出每位置 logit, 价值头用全局平均池化,
因此同一套权重可用于 9x9 到 19x19 的任意棋盘 (课程式训练的基础)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class PolicyValueNet(nn.Module):
    def __init__(self, channels=64, num_blocks=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])
        # 策略头: 每个位置输出一个 logit
        self.p_head = nn.Conv2d(channels, 1, 1)
        # 价值头: 全局平均池化 + MLP
        self.v_conv = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
        )
        self.v_fc1 = nn.Linear(2, 64)
        self.v_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        out = self.blocks(self.stem(x))
        p_logits = self.p_head(out).flatten(1)
        v = self.v_conv(out).mean(dim=(2, 3))
        v = torch.tanh(self.v_fc2(F.relu(self.v_fc1(v))))
        return p_logits, v

    def predict(self, state):
        """单个局面推理。state: (3, size, size) numpy 数组。
        返回 (策略概率 numpy, 价值 float)。"""
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            x = torch.from_numpy(state).unsqueeze(0).to(device)
            logits, v = self(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        return probs, float(v.item())


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    net = PolicyValueNet(ckpt["args"]["channels"], ckpt["args"]["blocks"]).to(device)
    net.load_state_dict(ckpt["model"])
    return net
