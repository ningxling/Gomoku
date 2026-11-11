"""定时从 VM 拉回最新权重，并报告训练进度（训练主力仍然在 VM 上）。

用法:
    python sync_vm.py                  # 查一次，有新权重就拉回来
    python sync_vm.py --loop 600       # 每 10 分钟查一次（可以一直挂着）
    python sync_vm.py --once --verbose

行为:
    1. ssh 上去读 checkpoints/latest.pt 的 mtime/大小 + train.log 最后一条 iter + 进程状态；
    2. 远端权重变了就 scp 到 checkpoints/latest.tmp，再原子替换成 checkpoints/latest.pt；
    3. 顺手把状态写进 checkpoints/vm_status.json，并在终端打一行进度；
    4. 连不上就重试 2 次，之后报错退出（--loop 模式下则等下一轮）。

server_all.py 会在每次用神经网络时检查 latest.pt 的 mtime，变了就自动重新加载，
所以不用重启服务器，网页直接就用上新权重。
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

HOST = "rocky@10.129.245.205"
REMOTE = "/home/rocky/gomoku"
HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_CKPT = os.path.join(HERE, "checkpoints", "latest.pt")
STATE = os.path.join(HERE, "checkpoints", "vm_state.json")
STATUS = os.path.join(HERE, "checkpoints", "vm_status.json")
HISTORY = os.path.join(HERE, "checkpoints", "history")

SSH_OPTS = ["-o", "ConnectTimeout=15", "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new"]

ITER_RE = re.compile(r"iter\s+(\d+)\s*\|\s*(\d+)x(\d+)\s*\|.*?loss\s+([\d.]+)")


def _run(args, timeout=90):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                          encoding="utf-8", errors="replace")


def ssh(cmd, timeout=90):
    return _run(["ssh", *SSH_OPTS, HOST, cmd], timeout=timeout)


def scp_pull(dst_tmp, timeout=300):
    return _run(["scp", *SSH_OPTS, f"{HOST}:{REMOTE}/checkpoints/latest.pt", dst_tmp],
                timeout=timeout)


def fetch_remote():
    """返回 dict(mtime, size, iter, size_board, loss, running, raw)。"""
    cmd = (
        f"cd {REMOTE} && "
        "stat -c '%Y %s' checkpoints/latest.pt; "
        "grep -a 'iter' train.log | tail -n 1; "
        "pgrep -af '^python3 train.py' | head -n 1"
    )
    r = ssh(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"ssh 失败: {r.stderr.strip()[:200]}")
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError("远端没有返回内容（checkpoints/train.log 可能不存在）")
    stat_parts = lines[0].split()
    info = {"mtime": int(stat_parts[0]), "size": int(stat_parts[1]),
            "iter": None, "board": None, "loss": None, "running": None}
    for ln in lines[1:]:
        m = ITER_RE.search(ln)
        if m:
            info.update(iter=int(m.group(1)),
                        board=f"{m.group(2)}x{m.group(3)}",
                        loss=float(m.group(4)))
        elif "train.py" in ln:
            info["running"] = True
    if info["running"] is None:
        info["running"] = False
    return info


def pull(local_path=LOCAL_CKPT, keep_history=5, verbose=True):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    tmp = local_path + ".tmp"
    for attempt in (1, 2):
        try:
            r = scp_pull(tmp)
            if r.returncode != 0:
                raise RuntimeError(r.stderr.strip()[:200])
            if os.path.getsize(tmp) < 10000:
                raise RuntimeError("拉回来的文件太小，可能不完整")
            os.replace(tmp, local_path)
            if keep_history > 0:
                os.makedirs(HISTORY, exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                hist = os.path.join(HISTORY, f"latest_{stamp}.pt")
                try:
                    with open(local_path, "rb") as f_in, open(hist, "wb") as f_out:
                        f_out.write(f_in.read())
                except OSError:
                    pass
                _prune_history(keep_history)
            return os.path.getsize(local_path)
        except Exception as e:                        # noqa: BLE001
            if verbose:
                print(f"  拉取失败({attempt}/2): {e}", flush=True)
            time.sleep(2)
    raise RuntimeError("拉取权重失败（重试 2 次）")


def _prune_history(keep):
    try:
        files = sorted(os.listdir(HISTORY))
    except OSError:
        return
    for name in files[:-keep] if len(files) > keep else []:
        try:
            os.remove(os.path.join(HISTORY, name))
        except OSError:
            pass


def load_state():
    try:
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:                                 # noqa: BLE001
        return {}


def save_state(state):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def report(info, fetched, local_size, err=None):
    ts = datetime.now().strftime("%H:%M:%S")
    if err:
        print(f"[{ts}] 连不上 VM: {err}", flush=True)
        return
    parts = [f"[{ts}]"]
    parts.append(f"第 {info['iter']} 轮" if info["iter"] is not None else "还没出第一轮")
    if info["board"]:
        parts.append(info["board"])
    if info["loss"] is not None:
        parts.append(f"loss {info['loss']:.3f}")
    parts.append("训练中" if info["running"] else "进程不在（可能已停）")
    if fetched:
        parts.append(f"→ 已回传最新权重 {local_size / 1024:.0f} KB")
    else:
        parts.append("权重无变化")
    print(" | ".join(parts), flush=True)


def run_bench(games):
    """回传成功后在本地打一轮基准赛（网络 vs 传统引擎），结果写 bench_log.csv。

    打完再把 CSV 推到 VM 一份——这样“哪个轮次多少胜率”的历史不只存在本地，
    即使本地电脑重装/会话没了，也能从 VM 上恢复。
    """
    script = os.path.join(HERE, "bench_vs_teacher.py")
    bench_log = os.path.join(HERE, "checkpoints", "bench_log.csv")
    if os.path.exists(script):
        try:
            r = subprocess.run([sys.executable, script, "--games", str(games),
                                "--size", "19", "--quiet"],
                               cwd=HERE, capture_output=True, text=True, timeout=1800,
                               encoding="utf-8", errors="replace")
            out = (r.stdout or "").strip()
            if out:
                print("  " + out.splitlines()[-1], flush=True)
            elif r.returncode != 0:
                print(f"  基准赛失败: {(r.stderr or '').strip()[:160]}", flush=True)
        except Exception as e:                        # noqa: BLE001
            print(f"  基准赛失败: {e}", flush=True)
    if os.path.exists(bench_log):
        try:
            _run(["scp", *SSH_OPTS, bench_log,
                  f"{HOST}:{REMOTE}/checkpoints/bench_log.csv"], timeout=120)
        except Exception as e:                        # noqa: BLE001
            print(f"  基准赛记录备份到 VM 失败: {e}", flush=True)


def one_round(args):
    state = load_state()
    info = fetch_remote()
    key = f"{info['mtime']}-{info['size']}"
    changed = state.get("key") != key
    local_size = os.path.getsize(LOCAL_CKPT) if os.path.exists(LOCAL_CKPT) else 0
    if changed:
        local_size = pull(LOCAL_CKPT, keep_history=args.keep_history)
        state = {"key": key, "iter": info["iter"], "board": info["board"],
                 "loss": info["loss"], "pulled_at": datetime.now().isoformat(timespec="seconds"),
                 "local_size": local_size}
        save_state(state)
    report(info, changed, local_size)
    if changed and args.bench > 0:
        run_bench(args.bench)
    status = dict(info)
    status["fetched_at"] = datetime.now().isoformat(timespec="seconds")
    status["local_size"] = local_size
    status["pulled_now"] = changed
    os.makedirs(os.path.dirname(STATUS), exist_ok=True)
    with open(STATUS, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)
    return 0


def main():
    global HOST, REMOTE
    ap = argparse.ArgumentParser(description="从 VM 定时回传训练权重")
    ap.add_argument("--loop", type=int, default=0, metavar="SEC",
                    help="每隔 SEC 秒查一次；0=只查一次")
    ap.add_argument("--once", action="store_true", help="只查一次（默认）")
    ap.add_argument("--keep-history", type=int, default=5, help="本地保留几份历史权重")
    ap.add_argument("--bench", type=int, default=0,
                    help="回传成功后顺便和传统引擎打 N 局基准赛（0=不打）")
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--remote", default=REMOTE)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    HOST, REMOTE = args.host, args.remote

    if args.loop <= 0 or args.once:
        for attempt in (1, 2, 3):
            try:
                return one_round(args)
            except Exception as e:                     # noqa: BLE001
                if attempt >= 3:
                    print(f"连不上 VM（已重试 2 次）: {e}")
                    return 2
                time.sleep(3)

    print(f"开始定时同步：每 {args.loop} 秒检查一次，Ctrl+C 结束", flush=True)
    while True:
        try:
            one_round(args)
        except Exception as e:                         # noqa: BLE001
            print(f"[{datetime.now():%H:%M:%S}] 本轮失败: {e}", flush=True)
        time.sleep(max(30, args.loop))


if __name__ == "__main__":
    sys.exit(main())
