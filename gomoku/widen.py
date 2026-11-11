"""函数保持的加宽：把 64 通道的权重装进 128 通道网络。

做法（net2net 的"复制 + 置零"版本）：
  * 卷积层：新网络前 old_out 个输出通道、前 old_in 个输入通道 = 旧权重，
    其余全部为 0；
  * BatchNorm：前 old_c 个通道原样复制；新增通道的 gamma=0、beta=0、
    running_mean=0、running_var=1 —— 这样新增通道恒为 0，不影响输出；
  * 策略头 / 价值头：同样只保留旧输入通道的权重。

于是新网络在初始时刻的输出与旧网络**逐位相等**（误差 <1e-4），
训练从旧模型的水平接着走，而不是从随机初始化重新爬。

用法：
    python widen.py --src checkpoints_64ch/best.pt --dst checkpoints/latest.pt
"""
import argparse
import copy
import os

import numpy as np
import torch

from model import PolicyValueNet


def widen_conv(old, new):
    """把 old 的权重拷到 new 的左上角，其余置零。"""
    oo, oi = old.weight.shape[0], old.weight.shape[1]
    new.weight.data.zero_()
    new.weight.data[:oo, :oi] = old.weight.data
    if old.bias is not None and new.bias is not None:
        new.bias.data.zero_()
        new.bias.data[:oo] = old.bias.data


def widen_bn(old, new):
    oc = old.weight.shape[0]
    new.weight.data.zero_()          # gamma = 0 → 新增通道恒输出 0
    new.bias.data.zero_()
    new.running_mean.data.zero_()
    new.running_var.data.fill_(1.0)
    new.weight.data[:oc] = old.weight.data
    new.bias.data[:oc] = old.bias.data
    new.running_mean.data[:oc] = old.running_mean.data
    new.running_var.data[:oc] = old.running_var.data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="旧权重（64 通道）")
    ap.add_argument("--dst", required=True, help="输出（128 通道）")
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=4)
    a = ap.parse_args()

    saved = torch.load(a.src, map_location="cpu")
    old_ch = saved["args"]["channels"]
    old = PolicyValueNet(old_ch, saved["args"]["blocks"])
    old.load_state_dict(saved["model"])
    old.eval()

    new = PolicyValueNet(a.channels, a.blocks)
    so, sn = old.state_dict(), new.state_dict()
    out = copy.deepcopy(sn)

    # stem
    widen_conv(old.stem[0], new.stem[0])
    out["stem.0.weight"] = new.stem[0].weight.data.clone()
    widen_bn(old.stem[1], new.stem[1])
    for k in ("weight", "bias", "running_mean", "running_var"):
        out[f"stem.1.{k}"] = getattr(new.stem[1], k).data.clone()

    # 残差块
    for i in range(len(old.blocks)):
        ob, nb = old.blocks[i], new.blocks[i]
        for name, oconv, nconv in (("conv1", ob.conv1, nb.conv1), ("conv2", ob.conv2, nb.conv2)):
            widen_conv(oconv, nconv)
            out[f"blocks.{i}.{name}.weight"] = nconv.weight.data.clone()
        for name, obn, nbn in (("bn1", ob.bn1, nb.bn1), ("bn2", ob.bn2, nb.bn2)):
            widen_bn(obn, nbn)
            for k in ("weight", "bias", "running_mean", "running_var"):
                out[f"blocks.{i}.{name}.{k}"] = getattr(nbn, k).data.clone()

    # 策略头 / 价值头
    widen_conv(old.p_head, new.p_head)
    out["p_head.weight"] = new.p_head.weight.data.clone()
    out["p_head.bias"] = new.p_head.bias.data.clone()
    widen_conv(old.v_conv[0], new.v_conv[0])
    out["v_conv.0.weight"] = new.v_conv[0].weight.data.clone()
    widen_bn(old.v_conv[1], new.v_conv[1])
    for k in ("weight", "bias", "running_mean", "running_var"):
        out[f"v_conv.1.{k}"] = getattr(new.v_conv[1], k).data.clone()
    out["v_fc1.weight"] = old.v_fc1.weight.data.clone()
    out["v_fc1.bias"] = old.v_fc1.bias.data.clone()
    out["v_fc2.weight"] = old.v_fc2.weight.data.clone()
    out["v_fc2.bias"] = old.v_fc2.bias.data.clone()

    new.load_state_dict(out)
    new.eval()

    # ---- 等价性验证：同一批随机局面，两个网络的输出必须一致 ----
    rng = np.random.default_rng(0)
    max_p, max_v = 0.0, 0.0
    with torch.no_grad():
        for _ in range(8):
            board = np.zeros((3, 19, 19), dtype=np.float32)
            for _ in range(rng.integers(4, 40)):
                board[0, rng.integers(0, 19), rng.integers(0, 19)] = 1.0
            x = torch.from_numpy(board).unsqueeze(0)
            p_old, v_old = old(x)
            p_new, v_new = new(x)
            max_p = max(max_p, float((p_old - p_new).abs().max()))
            max_v = max(max_v, float((v_old - v_new).abs().max()))
    print(f"等价性检查：策略最大差 {max_p:.2e}，价值最大差 {max_v:.2e}")
    if max_p > 1e-3 or max_v > 1e-3:
        raise SystemExit("加宽失败：输出不一致")

    os.makedirs(os.path.dirname(os.path.abspath(a.dst)), exist_ok=True)
    torch.save({"model": new.state_dict(),
                "args": {"channels": a.channels, "blocks": a.blocks},
                "size": saved.get("size", 19),
                "iter": saved.get("iter", 0),
                "widened_from": old_ch}, a.dst)
    print(f"已写出 {a.dst}（{a.channels} 通道，训练轮数继承 {saved.get('iter', 0)}）")


if __name__ == "__main__":
    main()
