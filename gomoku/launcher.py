"""打包入口（双引擎版）：启动服务器 + 自动打开浏览器。

PyInstaller 用这个文件当入口，而不是 play_all.py——因为 play_all.py 是用
subprocess 再起一个 python 进程，打包成 exe 后没有 python 可执行文件，
必须改成"同进程直接调用 server_all.main()"。
"""
import sys
import threading
import time
import urllib.request
import webbrowser

import server_all

PORT = 8000
if "--port" in sys.argv:
    try:
        PORT = int(sys.argv[sys.argv.index("--port") + 1])
    except (IndexError, ValueError):
        PORT = 8000


def open_when_ready():
    """等服务器起来再开浏览器，避免打开一个空白错误页。"""
    for _ in range(600):
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/heartbeat", timeout=1)
            webbrowser.open(f"http://localhost:{PORT}")
            return
        except Exception:
            time.sleep(0.5)


if __name__ == "__main__":
    threading.Thread(target=open_when_ready, daemon=True).start()
    server_all.main()
