"""一键启动路线 1（传统 α-β 搜索引擎）版五子棋。

用法:
    python play_ab.py                 # 默认每步 1.2 秒
    python play_ab.py --time 3        # 更强
    python play_ab.py --no-forbidden  # 关闭禁手

也可以直接双击项目根目录的 “开始游戏_传统AI.bat”。
网页用的还是 kimi3 那份 gomoku.html，一行没改。
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
    args = sys.argv[1:] or ["--time", "1.2"]
    subprocess.run([sys.executable, "server_ab.py"] + args)
