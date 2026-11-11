"""一键启动“两种 AI 可切换”的五子棋：开服务器 + 自动打开浏览器。

用法:
    python play_all.py                 # 传统引擎每步 1.2 秒，神经网络每步 400 次模拟
    python play_all.py --time 2        # 传统引擎更强
    python play_all.py --sims 800      # 神经网络更强（更慢）
    python play_all.py --ckpt checkpoints/pretrain.pt

也可以直接双击项目根目录的 “开始游戏.bat”。
"""
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

PORT = 8000


def open_when_ready():
    for _ in range(240):
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/heartbeat", timeout=1)
            webbrowser.open(f"http://localhost:{PORT}")
            return
        except Exception:
            time.sleep(0.5)


if __name__ == "__main__":
    threading.Thread(target=open_when_ready, daemon=True).start()
    args = sys.argv[1:]
    subprocess.run([sys.executable, "server_all.py"] + args)
