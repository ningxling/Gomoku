"""打包入口（轻量版）：只有传统引擎，不需要 torch / numpy。

体积小（约 15 MB）、启动快（1~2 秒），适合发给没装 Python 的人直接玩。
"""
import sys
import threading
import time
import urllib.request
import webbrowser

import server_ab

PORT = 8000
if "--port" in sys.argv:
    try:
        PORT = int(sys.argv[sys.argv.index("--port") + 1])
    except (IndexError, ValueError):
        PORT = 8000


def open_when_ready():
    for _ in range(600):
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/heartbeat", timeout=1)
            webbrowser.open(f"http://localhost:{PORT}")
            return
        except Exception:
            time.sleep(0.5)


if __name__ == "__main__":
    threading.Thread(target=open_when_ready, daemon=True).start()
    server_ab.main()
