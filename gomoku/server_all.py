"""一个服务器同时提供两种 AI，网页上直接切换：

    ab  = 路线 1 传统搜索（α-β + 棋型评估 + VCF 算杀，纯 Python，不需要 torch）
    nn  = kimi3 的神经网络（AlphaZero 风格 MCTS + checkpoints/latest.pt）

接口（与 kimi3 的 server.py 兼容，多认两个可选字段）：

    GET  /                返回 ../gomoku.html
    GET  /heartbeat       心跳，5 分钟没动静自动退出（后台标签页会被浏览器节流）
    GET  /info            报告两个引擎是否可用（网页据此灰掉不可用的选项）
    POST /move            {"board": [[0,1,-1,...], ...], "player": -1,
                           "engine": "ab" | "nn",     ← 不传默认 ab
                           "forbidden": true}          ← 不传默认 true
                          -> {"move": 12, "row": 1, "col": 3, "engine": "ab", "info": {...}}

用法:
    python server_all.py                  # 自动识别棋盘大小，传统引擎每步 1.2 秒
    python server_all.py --time 2         # 传统引擎更强
    python server_all.py --sims 400       # 神经网络每步模拟次数更多（更强也更慢）
    python server_all.py --ckpt checkpoints/pretrain.pt
"""
import argparse
import importlib.util
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from engine import Engine

HERE = os.path.dirname(os.path.abspath(__file__))
# PyInstaller 打包后，数据和代码被解到 sys._MEIPASS；源码运行时没有这个变量
BUNDLE = getattr(sys, "_MEIPASS", HERE)


def _res(*parts):
    """找资源文件：先看打包目录，再按源码目录布局（本目录 / 上一级）找。"""
    for base in (BUNDLE, HERE, os.path.join(HERE, "..")):
        p = os.path.join(base, *parts)
        if os.path.exists(p):
            return p
    return os.path.join(BUNDLE, *parts)


HTML_PATH = _res("gomoku.html")
HEARTBEAT_TIMEOUT = 300          # 秒；浏览器节流后台标签页时心跳可能延迟到 60s+，留足余量


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


def probe_nn(ckpt):
    """不真正加载，只判断“神经网络这条路能不能走”。"""
    if not os.path.exists(ckpt):
        return False, f"找不到权重文件 {os.path.basename(ckpt)}"
    for mod, name in (("torch", "torch"), ("numpy", "numpy")):
        if importlib.util.find_spec(mod) is None:
            return False, f"没有安装 {name}"
    return True, ""


class _NNError(Exception):
    pass


class _NN:
    """懒加载神经网络。定时同步脚本把 latest.pt 换掉后，下一次用就自动重新加载。"""

    def __init__(self, ckpt):
        self.ckpt = ckpt
        self.ready = False
        self.error = ""
        self.mtime = -1.0
        self.net = None
        self.mcts = None
        self.game_cls = None
        self.np = None

    def load(self):
        try:
            mtime = os.path.getmtime(self.ckpt)
        except OSError:
            mtime = -1.0
        if mtime == self.mtime and (self.ready or self.error):
            return self.ready                     # 没变过，沿用上次结果
        self.ready = False
        self.error = ""
        self.mtime = mtime
        if mtime < 0:
            self.error = f"找不到权重文件 {os.path.basename(self.ckpt)}"
            return False
        try:
            import numpy as np
            import torch
            from game import GomokuGame
            from mcts import MCTS
            from model import PolicyValueNet

            # 单次 19x19 前向很小，线程开太多反而互相拖累（实测 4 线程最优）
            torch.set_num_threads(min(4, os.cpu_count() or 4))
            saved = torch.load(self.ckpt, map_location="cpu")
            a = saved.get("args", {})
            net = PolicyValueNet(a.get("channels", 64), a.get("blocks", 4))
            net.load_state_dict(saved["model"])
            net.eval()
            self.np = np
            self.net = net
            self.mcts = MCTS(net)
            self.game_cls = GomokuGame
            self.ready = True
            # 预热：把 torch 的卷积核初始化、内存池和 MCTS 的代码路径先跑一遍，
            # 否则"第一步"要额外多花 7~8 秒（实测 12.5s vs 后续 5s）
            warm = GomokuGame(19, 5, True)
            for _ in range(3):
                net.predict(warm.encode())
            self.mcts.get_policy(warm, 40, temperature=0.0, add_noise=False)
            print(f"已加载神经网络权重 {self.ckpt}（训练轮数 {saved.get('iter', '?')}）",
                  flush=True)
        except Exception as e:                      # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            print(f"神经网络不可用: {self.error}", flush=True)
        return self.ready


class Handler(BaseHTTPRequestHandler):
    ab = None                 # engine.Engine
    nn = None                 # _NN
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
            nn_ok, nn_reason = probe_nn(self.cfg.get("ckpt", ""))
            if self.nn is not None and self.nn.error and self.nn.mtime >= 0:
                nn_ok, nn_reason = False, self.nn.error
            self._json({
                "engines": {"ab": True, "nn": bool(nn_ok)},
                "nn_reason": nn_reason,
                "size": self.cfg.get("size", 0),
                "ab_time": self.cfg.get("time", 1.2),
                "sims": self.cfg.get("sims", 400),
                "ckpt": self.cfg.get("ckpt", ""),
                "ckpt_mtime": self.cfg.get("ckpt_mtime", 0),
            })
        else:
            self._json({"error": "unknown path"}, 404)

    # ---------------- POST /move ---------------- #
    def do_POST(self):
        if self.path != "/move":
            return self._json({"error": "unknown path"}, 404)
        try:
            Handler.last_heartbeat = time.time()
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            board2d = data["board"]
            size = len(board2d)
            if size < 5 or any(len(row) != size for row in board2d):
                return self._json({"error": "棋盘必须是 size x size 的方阵"}, 400)
            want = self.cfg.get("size", 0)
            if want and size != want:
                return self._json({"error": f"服务器配置为 {want}x{want} 棋盘"}, 400)

            player = 1 if int(data["player"]) == 1 else -1
            forbidden = bool(data.get("forbidden", True)) and not self.cfg.get("no_forbidden")
            engine = str(data.get("engine", "ab")).lower()
            if engine not in ("ab", "nn"):
                return self._json({"error": f"未知引擎 {engine!r}"}, 400)
            if engine == "nn" and not probe_nn(self.cfg.get("ckpt", ""))[0]:
                return self._json(
                    {"error": f"神经网络不可用: {probe_nn(self.cfg.get('ckpt', ''))[1]}"}, 503)

            if board_full(board2d):
                return self._json({"error": "可落子点已满"}, 400)

            t0 = time.perf_counter()
            try:
                with self.lock:
                    if engine == "ab":
                        move, info = self._move_ab(board2d, size, player, forbidden)
                    else:
                        move, info = self._move_nn(board2d, size, player, forbidden)
            except _NNError as e:
                return self._json({"error": f"神经网络不可用: {e}"}, 503)
            if move is None:
                return self._json({"error": "无合法着法"}, 400)

            info = {"reason": info.get("reason", ""), "depth": info.get("depth", 0),
                    "nodes": info.get("nodes", 0), "sims": info.get("sims", 0),
                    "think": round(time.perf_counter() - t0, 3)}
            print(f"[{engine}/{ '黑' if player == 1 else '白' }] {info['reason']} "
                  f"depth={info['depth']} nodes={info['nodes']} "
                  f"{info['think']}s -> ({move // size},{move % size})", flush=True)
            self._json({"move": move, "row": move // size, "col": move % size,
                        "engine": engine, "info": info})
        except Exception as e:                      # noqa: BLE001
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    # ---------------- 两种引擎 ---------------- #
    def _move_ab(self, board2d, size, player, forbidden):
        cells = to_cells(board2d, size)
        self.ab.forbidden = forbidden
        self.ab.set_position(size, cells, 1 if player == 1 else 2)
        if not any(cells):
            return size // 2 + (size // 2) * size, {"reason": "开局天元"}
        return self.ab.choose_move()

    def _move_nn(self, board2d, size, player, forbidden):
        nn = self.nn
        if not nn.load():
            raise _NNError(nn.error)
        np = nn.np
        game = nn.game_cls(size, 5, forbidden)
        game.board = np.array(board2d, dtype=np.int8)
        game.current_player = player
        if not game.legal_moves():
            return None, {}
        pi = nn.mcts.get_policy(game, self.cfg.get("sims", 400),
                                temperature=0.0, add_noise=False)
        return int(pi.argmax()), {"reason": "MCTS + 策略价值网络",
                                  "sims": self.cfg.get("sims", 400)}

    def log_message(self, fmt, *args):
        pass


def board_full(board2d):
    for row in board2d:
        for v in row:
            if v == 0:
                return False
    return True


def main():
    ap = argparse.ArgumentParser(description="五子棋 AI 服务器（传统搜索 + 神经网络，网页可切换）")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1",
                    help="监听地址：127.0.0.1 只有本机能访问；0.0.0.0 允许局域网/公网访问")
    ap.add_argument("--time", type=float, default=1.2, help="传统引擎每步思考秒数")
    ap.add_argument("--depth", type=int, default=8, help="传统引擎最大搜索层数")
    ap.add_argument("--vcf-depth", type=int, default=6, help="传统引擎 VCF 算杀深度")
    ap.add_argument("--sims", type=int, default=200,
                    help="神经网络每步 MCTS 模拟次数（19x19 上每次模拟约 18ms，200 次≈3.7 秒）")
    ap.add_argument("--size", type=int, default=0, help="固定棋盘大小，0=自动跟随网页")
    ap.add_argument("--no-forbidden", action="store_true", help="强制关闭禁手（覆盖网页选项）")
    ap.add_argument("--ckpt", default=(_res("checkpoints", "best.pt")
                                       if os.path.exists(_res("checkpoints", "best.pt"))
                                       else _res("checkpoints", "latest.pt")),
                    help="权重路径；默认优先用闸门选出的 best.pt")
    args = ap.parse_args()

    Handler.ab = Engine(time_limit=args.time, max_depth=args.depth,
                        forbidden=not args.no_forbidden, vcf_depth=args.vcf_depth)
    Handler.nn = _NN(args.ckpt)
    nn_ok, nn_reason = probe_nn(args.ckpt)
    try:
        ckpt_mtime = int(os.path.getmtime(args.ckpt))
    except OSError:
        ckpt_mtime = 0
    Handler.cfg = {"size": args.size, "time": args.time, "sims": args.sims,
                   "no_forbidden": args.no_forbidden, "nn_ok": nn_ok,
                   "nn_reason": nn_reason, "ckpt": args.ckpt,
                   "ckpt_mtime": ckpt_mtime}

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)

    # 启动时就把网络权重加载 + 预热好，否则第一次走神经网络要等 10 秒以上
    if nn_ok:
        print("正在加载神经网络权重…", flush=True)
        t0 = time.time()
        if Handler.nn.load():
            print(f"神经网络就绪（耗时 {time.time() - t0:.1f}s）", flush=True)
        else:
            print(f"神经网络加载失败: {Handler.nn.error}", flush=True)

    def watchdog():
        while True:
            time.sleep(10)
            if Handler.last_heartbeat and \
                    time.time() - Handler.last_heartbeat > HEARTBEAT_TIMEOUT:
                print("网页已关闭，服务器自动退出")
                threading.Thread(target=httpd.shutdown, daemon=True).start()
                return

    threading.Thread(target=watchdog, daemon=True).start()
    shown = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    print(f"AI 服务器已启动: http://{shown}:{args.port}"
          + ("（局域网可访问）" if args.host == "0.0.0.0" else ""))
    print(f"  传统搜索 (ab): 每步 {args.time}s, 禁手 {'开' if not args.no_forbidden else '关'}")
    if nn_ok:
        print(f"  神经网络 (nn): {args.ckpt}, 每步 {args.sims} 次模拟")
    else:
        print(f"  神经网络 (nn): 不可用（{nn_reason}），网页上会灰掉")
    print("浏览器打开上面的地址，在“AI 引擎”里切换两种模式；关闭网页后服务器自动退出")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
