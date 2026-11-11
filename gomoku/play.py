"""一键启动: 开 AI 服务器 + 自动打开浏览器。网页关闭后服务器自动退出。

用法: python play.py [--size 19] [--sims 400]
也可以直接双击项目根目录的 "开始游戏.bat"。
"""
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

PORT = 8000


def open_when_ready():
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/heartbeat", timeout=1)
            webbrowser.open(f"http://localhost:{PORT}")
            return
        except Exception:
            time.sleep(0.5)


if __name__ == "__main__":
    threading.Thread(target=open_when_ready, daemon=True).start()
    args = sys.argv[1:] or ["--size", "19"]
    subprocess.run([sys.executable, "server.py"] + args)
