"""在 VM 上定时给 checkpoints/latest.pt 做快照，并记录每份权重对应的训练信息。

为什么放在 VM 上：本地电脑不一定一直开着，但训练机一直在跑。这样以后想做
"训练 500 轮 vs 训练 3000 轮，谁更强" 的对比时，直接拿存档里的 .pt 就能打。

用法（在 /home/rocky/gomoku 下）:
    python3 snapshot_ckpt.py --once            # 打一次快照
    python3 snapshot_ckpt.py --interval 1800   # 每 30 分钟一次（默认）
    python3 snapshot_ckpt.py --keep 400        # 最多保留多少份（默认 400，约 480 MB）

产出:
    checkpoints/archive/iter00504_19x19_20261020_2130.pt
    checkpoints/archive/index.csv     # 时间,轮数,棋盘,loss,字节,训练命令

快照是"复制 + 校验 + 改名"，不会碰训练进程正在写的 latest.pt；
如果恰好读到写了一半的文件（校验不过），会等几秒重试。
"""
import argparse
import csv
import os
import shutil
import time
import zipfile
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "checkpoints", "latest.pt")
ARCHIVE = os.path.join(HERE, "checkpoints", "archive")
INDEX = os.path.join(ARCHIVE, "index.csv")
LOG = os.path.join(HERE, "train.log")


def last_iter_info():
    """从 train.log 里读最后一条 iter 行 -> (iter, '19x19', loss)。"""
    if not os.path.exists(LOG):
        return None
    try:
        with open(LOG, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 65536))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    for line in reversed(tail.splitlines()):
        if "iter" in line and "loss" in line:
            try:
                parts = [p.strip() for p in line.split("|")]
                it = int(parts[0].split()[-1])
                board = parts[1]
                loss = float(parts[-2].split("loss")[-1].split("(")[0].strip())
                return it, board, loss
            except (ValueError, IndexError):
                return None
    return None


def train_cmdline():
    """找到正在跑的 train.py，把命令行记下来（这就是这版权重的训练配方）。"""
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                raw = f.read().decode("utf-8", "replace")
        except OSError:
            continue
        parts = [p for p in raw.split("\0") if p]
        if parts and parts[0].endswith("python3") and any(p.endswith("train.py") for p in parts):
            return " ".join(parts[1:]) if len(parts) > 1 else " ".join(parts)
    return ""


def looks_ok(path):
    """粗校验：torch.save 出来的现在是 zip 包，写一半的话解不开。"""
    try:
        if os.path.getsize(path) < 100_000:
            return False
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as z:
                return z.testzip() is None
        return True          # 老格式（非 zip），只能靠大小判断
    except Exception:         # noqa: BLE001
        return False


def read_last_index_iter():
    if not os.path.exists(INDEX):
        return None
    try:
        with open(INDEX, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        return int(rows[-1][1]) if len(rows) > 1 else None
    except Exception:         # noqa: BLE001
        return None


def _iter_of(name):
    try:
        return int(name[4:].split("_", 1)[0])
    except (ValueError, IndexError):
        return None


def prune(keep):
    """保留最新 keep 份，另外每个"千轮"桶留第一份（长期对比用，不会被清掉）。"""
    try:
        files = sorted(n for n in os.listdir(ARCHIVE) if n.endswith(".pt"))
    except OSError:
        return
    if keep <= 0 or len(files) <= keep:
        return
    milestone = {}
    for name in files:                 # files 是排好序的，桶内第一个就是最早的
        it = _iter_of(name)
        if it is not None:
            milestone.setdefault(it // 1000, name)
    protected = set(milestone.values())
    for name in files[:-keep]:
        if name in protected:
            continue
        try:
            os.remove(os.path.join(ARCHIVE, name))
        except OSError:
            pass


def snapshot_once(keep=400, verbose=True):
    if not os.path.exists(CKPT):
        return False
    info = last_iter_info()
    it, board, loss = (info if info else (None, "?", None))
    if it is not None and it == read_last_index_iter():
        return False          # 同一轮已经存过，不重复
    os.makedirs(ARCHIVE, exist_ok=True)
    tmp = os.path.join(ARCHIVE, ".tmp.pt")
    for attempt in (1, 2, 3):
        try:
            shutil.copy2(CKPT, tmp)
            if looks_ok(tmp):
                stamp = datetime.now().strftime("%Y%m%d_%H%M")
                name = f"iter{it if it is not None else 0:05d}_{board}_{stamp}.pt"
                os.replace(tmp, os.path.join(ARCHIVE, name))
                new_file = not os.path.exists(INDEX)
                with open(INDEX, "a", newline="", encoding="utf-8") as f:
                    wr = csv.writer(f)
                    if new_file:
                        wr.writerow(["时间", "轮数", "棋盘", "loss", "字节", "训练命令"])
                    wr.writerow([datetime.now().strftime("%Y-%m-%d %H:%M"), it, board,
                                 f"{loss:.3f}" if loss is not None else "",
                                 os.path.getsize(os.path.join(ARCHIVE, name)),
                                 train_cmdline()])
                prune(keep)
                if verbose:
                    print(f"[{datetime.now():%H:%M}] 已存档 {name} "
                          f"(loss {loss})", flush=True)
                return True
        except OSError as e:
            if verbose:
                print(f"  存档重试 {attempt}/3: {e}", flush=True)
        time.sleep(5)
    return False


def main():
    ap = argparse.ArgumentParser(description="给 latest.pt 定时做快照")
    ap.add_argument("--once", action="store_true", help="只打一次")
    ap.add_argument("--interval", type=int, default=1800, help="间隔秒数，默认 30 分钟")
    ap.add_argument("--keep", type=int, default=400, help="最多保留几份")
    args = ap.parse_args()

    if args.once:
        ok = snapshot_once(args.keep)
        print("完成" if ok else "没有新轮次，跳过")
        return 0
    print(f"开始定时存档：每 {args.interval} 秒检查一次，最多保留 {args.keep} 份", flush=True)
    while True:
        try:
            snapshot_once(args.keep)
        except Exception as e:                    # noqa: BLE001
            print(f"存档失败: {e}", flush=True)
        time.sleep(max(60, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
