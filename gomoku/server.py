"""AI 出招服务器：网页通过 HTTP 调用 AI。

用法:
    python server.py                        # 默认 9x9 棋盘, 有 checkpoint 自动加载
    python server.py --size 19 --sims 400   # 19x19 棋盘, 更强但每步更慢
    python server.py --no-forbidden         # 关闭禁手规则

接口:
    POST /move   {"board": [[0,1,-1,...], ...], "player": -1}
    返回 {"move": 12, "row": 1, "col": 3}

注意: --size 必须和网页上选择的棋盘大小一致。
"""
import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import torch

from game import GomokuGame
from mcts import MCTS
from model import PolicyValueNet

HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gomoku.html")
HEARTBEAT_TIMEOUT = 90  # 秒, 超过这个时间没收到网页心跳就自动关服务器


def make_ai(ckpt, size, sims, use_forbidden):
    net = PolicyValueNet().to("cpu")
    if os.path.exists(ckpt):
        saved = torch.load(ckpt, map_location="cpu")
        net = PolicyValueNet(saved["args"]["channels"], saved["args"]["blocks"]).to("cpu")
        net.load_state_dict(saved["model"])
        print(f"已加载权重 {ckpt} (训练轮数 {saved.get('iter', '?')})")
    else:
        print(f"未找到 {ckpt}, 使用随机初始化网络 (AI 会很弱, 请先在 VM 上训练并传回权重)")
    mcts = MCTS(net)
    return mcts, size, sims, use_forbidden


class Handler(BaseHTTPRequestHandler):
    ai = None  # (mcts, size, sims, use_forbidden)
    last_heartbeat = None

    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send_json({})

    def do_GET(self):
        if self.path in ("/", "/index.html", "/gomoku.html"):
            with open(HTML_PATH, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/heartbeat":
            Handler.last_heartbeat = time.time()
            self._send_json({"ok": True})
        else:
            self._send_json({"error": "unknown path"}, 404)

    def do_POST(self):
        if self.path != "/move":
            return self._send_json({"error": "unknown path"}, 404)
        try:
            Handler.last_heartbeat = time.time()
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            mcts, size, sims, use_forbidden = self.ai
            board = np.array(data["board"], dtype=np.int8)
            if board.shape != (size, size):
                return self._send_json({"error": f"服务器配置为 {size}x{size} 棋盘"}, 400)
            game = GomokuGame(size, use_forbidden=use_forbidden)
            game.board = board
            game.current_player = int(data["player"])
            if not game.legal_moves():
                return self._send_json({"error": "无可落子点"}, 400)
            pi = mcts.get_policy(game, sims, temperature=0, add_noise=False)
            move = int(pi.argmax())
            self._send_json({"move": move, "row": move // size, "col": move % size})
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def log_message(self, fmt, *args):
        pass  # 静默日志


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--sims", type=int, default=400, help="每步 MCTS 模拟次数, 越大越强越慢")
    ap.add_argument("--size", type=int, default=9, help="棋盘大小, 需与网页选择一致")
    ap.add_argument("--no-forbidden", action="store_true", help="关闭黑棋禁手规则")
    ap.add_argument("--ckpt", default="checkpoints/latest.pt")
    args = ap.parse_args()
    Handler.ai = make_ai(args.ckpt, args.size, args.sims, not args.no_forbidden)
    httpd = HTTPServer(("127.0.0.1", args.port), Handler)

    def watchdog():
        # 网页每 10 秒发一次心跳; 网页关闭后超时自动关服务器
        while True:
            time.sleep(10)
            if Handler.last_heartbeat and \
                    time.time() - Handler.last_heartbeat > HEARTBEAT_TIMEOUT:
                print("网页已关闭, 服务器自动退出")
                threading.Thread(target=httpd.shutdown).start()
                return

    threading.Thread(target=watchdog, daemon=True).start()
    print(f"AI 服务器已启动: http://localhost:{args.port}  "
          f"({args.size}x{args.size}, 每步模拟 {args.sims} 次, 禁手 {not args.no_forbidden})")
    print("浏览器打开上面的地址即可对战; 关闭网页后服务器会自动退出")
    httpd.serve_forever()
