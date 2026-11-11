"""路线 1（传统搜索）AI 服务器：与 kimi3 的网页版共用同一套 HTTP 协议。

直接替代 kimi3 的 server.py，网页 gomoku.html 一行都不用改：

    python server_ab.py                  # 自动识别棋盘大小，每步思考 1.2 秒
    python server_ab.py --time 2.5       # 每步 2.5 秒，更强
    python server_ab.py --size 15        # 固定 15x15（默认 auto，跟着网页选择走）
    python server_ab.py --no-forbidden   # 关闭禁手

接口（与 server.py 完全一致）:
    GET  /                返回 ../gomoku.html
    GET  /heartbeat       心跳（网页每 10 秒一次，90 秒没动静自动退出）
    POST /move            {"board": [[0,1,-1,...], ...], "player": -1}
                          -> {"move": 12, "row": 1, "col": 3, "info": {...}}

与 server.py 的区别:
    * 不需要 torch / 权重文件，纯 CPU，启动即用；
    * 棋盘大小从请求里自动推断，网页上切换 9/15/19 不需要重启服务器；
    * 返回里多一个 info 字段（搜索深度/节点数/耗时/着法类型），网页会忽略它。
"""
import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from engine import Engine

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = getattr(__import__("sys"), "_MEIPASS", HERE)   # PyInstaller 打包后的解包目录


def _res(*parts):
    for base in (BUNDLE, HERE, os.path.join(HERE, "..")):
        p = os.path.join(base, *parts)
        if os.path.exists(p):
            return p
    return os.path.join(BUNDLE, *parts)


HTML_PATH = _res("gomoku.html")
HEARTBEAT_TIMEOUT = 90


def to_cells(board2d, size):
    """网页的 0/1/-1 -> 引擎的 0/1/2。"""
    cells = [0] * (size * size)
    for r in range(size):
        row = board2d[r]
        for c in range(size):
            v = row[c]
            if v == 1:
                cells[r * size + c] = 1
            elif v == -1:
                cells[r * size + c] = 2
            elif v != 0:
                raise ValueError(f"非法棋子值 {v!r}")
    return cells


def board_full(cells):
    return all(v != 0 for v in cells)


class Handler(BaseHTTPRequestHandler):
    engine = None
    cfg = {}
    last_heartbeat = None
    lock = threading.Lock()

    # ---------------- 基础响应 ---------------- #
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._json({})

    # ---------------- GET ---------------- #
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
            self._json({"ok": True})
        elif self.path == "/info":
            self._json({"engine": "alpha-beta", **self.cfg})
        else:
            self._json({"error": "unknown path"}, 404)

    # ---------------- POST /move ---------------- #
    def do_POST(self):
        if self.path != "/move":
            return self._json({"error": "unknown path"}, 404)
        try:
            Handler.last_heartbeat = time.time()
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            data = json.loads(raw)
            board2d = data["board"]
            size = len(board2d)
            if size < 5 or any(len(row) != size for row in board2d):
                return self._json({"error": "棋盘必须是 size x size 的方阵"}, 400)
            want = self.cfg.get("size", 0)
            if want and size != want:
                return self._json({"error": f"服务器配置为 {want}x{want} 棋盘"}, 400)

            cells = to_cells(board2d, size)
            player = 1 if int(data["player"]) == 1 else 2
            if board_full(cells):
                return self._json({"error": "可落子点已满"}, 400)

            engine_name = str(data.get("engine", "ab")).lower()
            if engine_name == "nn":
                return self._json(
                    {"error": "本服务器只有传统引擎；要两种模式都选请运行 server_all.py"}, 400)
            if engine_name != "ab":
                return self._json({"error": f"未知引擎 {engine_name!r}"}, 400)
            # 网页上的“禁手”勾选框也会传给服务器，让 AI 和玩家用同一套规则
            forbidden = bool(data.get("forbidden", True)) and self.cfg.get("forbidden", True)

            with self.lock:
                self.engine.forbidden = forbidden
                self.engine.set_position(size, cells, player)
                if not any(cells):
                    # 空盘：引擎自己会走天元，这里省掉一次搜索
                    move = size // 2 + (size // 2) * size
                    info = {"reason": "开局天元", "depth": 0, "nodes": 0, "time": 0.0}
                else:
                    move, info = self.engine.choose_move()

            if move is None:
                return self._json({"error": "无合法着法"}, 400)
            info = {"reason": info.get("reason", ""), "depth": info.get("depth", 0),
                    "nodes": info.get("nodes", 0),
                    "think": round(info.get("time", 0.0), 3)}
            if self.cfg.get("verbose", True):
                print(f"[{player == 1 and '黑' or '白'}] {info['reason']} "
                      f"depth={info['depth']} nodes={info['nodes']} "
                      f"time={info['think']}s -> ({move // size},{move % size})", flush=True)
            self._json({"move": move, "row": move // size, "col": move % size, "info": info})
        except Exception as e:                      # noqa: BLE001
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def log_message(self, fmt, *args):
        pass


def main():
    ap = argparse.ArgumentParser(description="路线1（α-β 搜索）五子棋 AI 服务器")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--time", type=float, default=1.2, help="每步思考秒数，越大越强")
    ap.add_argument("--depth", type=int, default=8, help="迭代加深的最大层数")
    ap.add_argument("--size", type=int, default=0, help="固定棋盘大小，0=自动跟随网页")
    ap.add_argument("--no-forbidden", action="store_true", help="关闭黑棋禁手")
    ap.add_argument("--vcf-depth", type=int, default=6, help="VCF 算杀深度")
    ap.add_argument("--sims", type=int, default=None, help="（兼容参数，忽略）")
    ap.add_argument("--ckpt", default=None, help="（兼容参数，忽略）")
    ap.add_argument("--quiet", action="store_true", help="不在终端打印每步信息")
    args = ap.parse_args()

    engine = Engine(time_limit=args.time, max_depth=args.depth,
                    forbidden=not args.no_forbidden, vcf_depth=args.vcf_depth)
    Handler.engine = engine
    Handler.cfg = {"size": args.size, "time": args.time, "depth": args.depth,
                   "forbidden": not args.no_forbidden, "verbose": not args.quiet}
    if args.sims is not None or args.ckpt:
        print("提示: --sims / --ckpt 是神经网络版的参数，本服务器不需要，已忽略")

    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)

    def watchdog():
        while True:
            time.sleep(10)
            if Handler.last_heartbeat and \
                    time.time() - Handler.last_heartbeat > HEARTBEAT_TIMEOUT:
                print("网页已关闭，服务器自动退出")
                threading.Thread(target=httpd.shutdown, daemon=True).start()
                return

    threading.Thread(target=watchdog, daemon=True).start()
    print(f"路线1 AI 服务器已启动: http://localhost:{args.port}  "
          f"(每步 {args.time}s, 最大深度 {args.depth}, "
          f"禁手 {'开' if not args.no_forbidden else '关'})")
    print("用浏览器打开上面的地址即可对战；关闭网页后服务器会自动退出")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
