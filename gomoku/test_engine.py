"""路线 1 引擎自测：棋规、禁手、算杀、自对弈合法性、HTTP 协议。

运行:
    python test_engine.py
全部通过时退出码为 0。
"""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np

import engine as E
from teacher import TeacherAgent

HERE = os.path.dirname(os.path.abspath(__file__))
FAILS = []


def check(name, cond, extra=""):
    print(("  [OK]   " if cond else "  [FAIL] ") + name + (f"  {extra}" if extra else ""))
    if not cond:
        FAILS.append(name)


def board(size, blacks=(), whites=()):
    cells = [0] * (size * size)
    for r, c in blacks:
        cells[r * size + c] = 1
    for r, c in whites:
        cells[r * size + c] = 2
    return cells


def has_five(cells, mv, p, n):
    r, c = divmod(mv, n)
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        k = 1
        for s in (1, -1):
            rr, cc = r + s * dr, c + s * dc
            while 0 <= rr < n and 0 <= cc < n and cells[rr * n + cc] == p:
                k += 1
                rr += s * dr
                cc += s * dc
        if k >= 5:
            return True
    return False


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_rules():
    print("棋规与评估")
    e = E.Engine(time_limit=0.2)
    e.set_position(9, [0] * 81, 1)
    mv, info = e.choose_move()
    check("空盘走天元", mv == 40, f"move={mv}")

    cells = board(9, [(4, 1), (4, 2), (4, 3), (4, 4)], [(0, 0), (8, 8)])
    e.set_position(9, cells, 1)
    mv, info = e.choose_move()
    check("已有四必成五", has_five(cells, mv, 1, 9), f"move={divmod(mv, 9)}")

    cells = board(9, [(0, 0), (8, 8)], [(4, 1), (4, 2), (4, 3), (4, 4)])
    cells[4 * 9 + 0] = 1                     # 一头被堵，只剩一个挡点
    e.set_position(9, cells, 1)
    mv, info = e.choose_move()
    check("单头四必挡", mv == 4 * 9 + 5, f"move={divmod(mv, 9)} reason={info['reason']}")

    cells = board(9, [(4, 1), (4, 2), (4, 3), (4, 5)], [(0, 0), (0, 1), (0, 2)])
    e.set_position(9, cells, 1)
    mv, info = e.choose_move()
    check("跳四也算四（下一手成五）", has_five(cells, mv, 1, 9),
          f"move={divmod(mv, 9)} reason={info['reason']}")


def test_forbidden():
    print("禁手判定")
    e = E.Engine(time_limit=0.2)
    cells = board(9, [(4, 1), (4, 2), (4, 3), (4, 4), (4, 6)])
    e.set_position(9, cells, 1)
    check("长连禁手", not e.is_legal(4 * 9 + 5, 1))
    check("长连旁边的普通点可以下", e.is_legal(4 * 9 + 0, 1))

    cells = board(9, [(4, 2), (4, 3), (2, 4), (3, 4)])
    e.set_position(9, cells, 1)
    check("双活三禁手(横竖各一活三)", not e.is_legal(4 * 9 + 4, 1))
    check("白棋没有禁手", e.is_legal(4 * 9 + 4, 2))

    # 跳活三：横 (4,3)(4,4) 下在 (4,6) 会形成 3,4,_,6，补 5 就是活四；
    # 竖 (2,6)(3,6) 下在 (4,6) 是普通活三。严格版应判禁手，网页的连续版判不出来。
    cells = board(9, [(4, 3), (4, 4), (2, 6), (3, 6)])
    e.set_position(9, cells, 1)
    check("跳三也算三（严格版禁手）", not e.is_legal(4 * 9 + 6, 1))
    check("严格版比网页版更严（网页会放行这一点）",
          not e._forbidden_quick(4 * 9 + 6))


def test_vcf():
    print("算杀 (VCF)")
    e = E.Engine(time_limit=0.3)
    cells = board(9, [(4, 3), (4, 4), (4, 5)], [(0, 0), (8, 8), (0, 8)])
    e.set_position(9, cells, 1)
    mv, info = e.choose_move()
    ok = info["reason"] == "VCF 算杀" or has_five(cells, mv, 1, 9)
    check("活三可算杀做活四", ok, f"move={divmod(mv, 9)} reason={info['reason']}")


def test_refute():
    print("防守：对方活三要挡")
    e = E.Engine(time_limit=0.5)
    # 白在第 0 列 5~7 做活三（两端 4 和 8 都空），黑必须去挡一端
    cells = board(9, [(4, 4), (3, 3)], [(0, 5), (0, 6), (0, 7)])
    e.set_position(9, cells, 1)
    mv, info = e.choose_move()
    check("挡边上活三", divmod(mv, 9) in ((0, 4), (0, 8)),
          f"move={divmod(mv, 9)} {info}")


def test_far_threat():
    print("远处的活三不能漏（19x19 回归用例）")
    e = E.Engine(time_limit=1.5)
    cells = board(19, [(9, 9), (10, 10), (10, 9)],
                  [(17, 0), (16, 0), (15, 0)])
    e.set_position(19, cells, 1)
    mv, info = e.choose_move()
    r, c = divmod(mv, 19)
    # 白 15~17 是活三，(13,0)/(14,0)/(18,0) 都能破掉“活四”威胁
    check("必须处理第 0 列的活三", c == 0 and r in (13, 14, 18),
          f"move={(r, c)} {info['reason']} depth={info['depth']}")


def test_selfplay():
    print("自对弈合法性（9x9 4 局 / 15x15 2 局）")
    bad = 0
    for size, games in ((9, 4), (15, 2)):
        e = E.Engine(time_limit=0.02, max_depth=6)
        for _ in range(games):
            cells = [0] * (size * size)
            turn = 1
            for _ply in range(size * size):
                e.set_position(size, cells, turn)
                mv, info = e.choose_move()
                if mv is None:
                    break
                if cells[mv] != 0 or (turn == 1 and not e.is_legal(mv, 1)):
                    bad += 1
                    break
                cells[mv] = turn
                if has_five(cells, mv, turn, size):
                    break
                turn = 3 - turn
    check("没有非法/禁手着法", bad == 0, f"bad={bad}")


def test_teacher():
    print("老师接口（给训练用）")
    t = TeacherAgent(time_limit=0.05, temperature=0.3, topk=4)
    e = E.Engine(time_limit=0.05)
    cells = board(9, [(4, 4), (4, 5)], [(3, 3)])
    e.set_position(9, cells, 1)
    mv, _ = e.choose_move()
    pi = e.root_policy(0.3, 4)
    check("软标签归一到 1", abs(sum(pi) - 1.0) < 1e-6, f"sum={sum(pi):.4f}")
    check("软标签包含引擎着法", pi[mv] > 0, f"p={pi[mv]:.3f}")


def start_server(script, *extra):
    """起一个服务器子进程，返回 (proc, base_url)。起不来返回 (None, base)。"""
    port = free_port()
    proc = subprocess.Popen([sys.executable, script, "--port", str(port), *extra],
                            cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    base = f"http://127.0.0.1:{port}"
    for _ in range(160):
        try:
            urllib.request.urlopen(base + "/heartbeat", timeout=1).read()
            return proc, base
        except Exception:
            time.sleep(0.25)
    proc.kill()
    return None, base


def stop_server(proc):
    if proc is None:
        return
    proc.kill()
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def post_json(base, payload, timeout=120):
    req = urllib.request.Request(base + "/move", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def test_http():
    print("HTTP 协议（与 kimi3 的 gomoku.html 对齐）")
    proc, base = start_server("server_ab.py", "--time", "0.1", "--quiet")
    try:
        check("传统服务器起来了", proc is not None)
        if proc is None:
            return

        html = urllib.request.urlopen(base + "/", timeout=5).read().decode("utf-8", "ignore")
        check("GET / 返回 kimi3 的网页", "五子棋" in html and "canvas" in html)
        check("网页里有两个引擎的下拉框", 'id="engine"' in html and "传统搜索" in html)

        b = [[0] * 9 for _ in range(9)]
        b[4][4] = 1
        r = post_json(base, {"board": b, "player": -1})
        ok = r.get("move") is not None and b[r["row"]][r["col"]] == 0
        check("POST /move 返回合法空点", ok, str(r))
        check("返回格式与 server.py 一致", set(r) >= {"move", "row", "col"}, str(sorted(r)))

        b2 = [[0] * 15 for _ in range(15)]
        b2[7][7] = 1
        r2 = post_json(base, {"board": b2, "player": -1})
        check("15x15 无需重启（大小自动识别）", r2.get("move") is not None, str(r2)[:90])

        code = 0
        try:
            post_json(base, {"board": b, "player": -1, "engine": "nn"})
        except urllib.error.HTTPError as e:
            code = e.code
        check("纯传统服务器会拒绝 engine=nn（不会假装是神经网络）", code == 400, f"{code}")
    finally:
        stop_server(proc)


def test_unified_server():
    print("双引擎服务器 server_all.py（网页上切换 ab / nn）")
    proc, base = start_server("server_all.py", "--time", "0.1", "--sims", "20")
    try:
        check("双引擎服务器起来了", proc is not None)
        if proc is None:
            return
        info = json.loads(urllib.request.urlopen(base + "/info", timeout=5).read())
        check("/info 报告两个引擎", info.get("engines", {}).get("ab") is True,
              str(info)[:120])
        nn_ok = bool(info.get("engines", {}).get("nn"))

        html = urllib.request.urlopen(base + "/", timeout=5).read().decode("utf-8", "ignore")
        check("GET / 返回改过的网页", 'id="engine"' in html)

        b = [[0] * 9 for _ in range(9)]
        b[4][4] = 1
        r = post_json(base, {"board": b, "player": -1, "engine": "ab",
                             "forbidden": True})
        check("engine=ab 走传统引擎", r.get("engine") == "ab" and r["info"].get("depth"),
              str(r)[:130])
        check("传统引擎给出搜索信息", r["info"].get("nodes", 0) > 0, str(r["info"]))

        r2 = post_json(base, {"board": b, "player": -1, "engine": "ab",
                              "forbidden": False})
        check("禁手开关可以按请求传", r2.get("move") is not None
              and b[r2["row"]][r2["col"]] == 0, str(r2)[:90])

        if nn_ok:
            r3 = post_json(base, {"board": b, "player": -1, "engine": "nn"})
            check("engine=nn 走神经网络", r3.get("engine") == "nn"
                  and b[r3["row"]][r3["col"]] == 0, str(r3)[:130])
            check("神经网络报告模拟次数", r3["info"].get("sims") == 20, str(r3["info"]))
        else:
            print("  （没有权重，跳过 nn 用例）")
    finally:
        stop_server(proc)

    # 没有权重时，nn 应被标记为不可用
    proc, base = start_server("server_all.py", "--ckpt", "checkpoints/_no_such.pt",
                              "--time", "0.1")
    try:
        if proc is not None:
            info = json.loads(urllib.request.urlopen(base + "/info", timeout=5).read())
            check("没有权重时 /info 标 nn 不可用",
                  info.get("engines", {}).get("nn") is False, str(info)[:120])
            code = 0
            try:
                post_json(base, {"board": [[0] * 9 for _ in range(9)], "player": -1,
                                 "engine": "nn"})
            except urllib.error.HTTPError as e:
                code = e.code
            check("没有权重时 engine=nn 返回 503", code == 503, f"{code}")
    finally:
        stop_server(proc)


def test_aivai_flow():
    print("AI 对战流程（网页 aivai 模式：传统执黑 / 强化执白 交替请求）")
    proc, base = start_server("server_all.py", "--time", "0.1", "--sims", "20")
    try:
        check("服务器起来了", proc is not None)
        if proc is None:
            return
        n = 9
        b = [[0] * n for _ in range(n)]
        turn = 1
        engines = {1: "ab", -1: "nn"}
        moves = 0
        ended = False
        for _ply in range(n * n):
            r = post_json(base, {"board": b, "player": turn,
                                 "engine": engines[turn], "forbidden": True})
            if r.get("engine") != engines[turn] or "error" in r:
                check(f"第 {moves + 1} 手走对了引擎", False, str(r)[:120])
                return
            if b[r["row"]][r["col"]] != 0:
                check("落子都在空点", False, str(r)[:120])
                return
            b[r["row"]][r["col"]] = 1 if turn == 1 else -1
            moves += 1
            if has_five_signed(b, turn, n):
                ended = True
                break
            turn = -turn
        check("两种引擎交替下完一局（无报错、无非法落子）", moves >= 5,
              f"{moves} 手, {'终局' if ended else '未终局'}")
        check("AI 对战里确实换了引擎", moves > 1)
    finally:
        stop_server(proc)


def has_five_signed(b, p, n):
    for r in range(n):
        for c in range(n):
            if b[r][c] != p:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                k = 1
                for sg in (1, -1):
                    rr, cc = r + sg * dr, c + sg * dc
                    while 0 <= rr < n and 0 <= cc < n and b[rr][cc] == p:
                        k += 1
                        rr += sg * dr
                        cc += sg * dc
                if k >= 5:
                    return True
    return False


if __name__ == "__main__":
    test_rules()
    test_forbidden()
    test_vcf()
    test_refute()
    test_far_threat()
    test_selfplay()
    test_teacher()
    test_http()
    test_unified_server()
    test_aivai_flow()
    print()
    if FAILS:
        print(f"失败 {len(FAILS)} 项: {FAILS}")
        sys.exit(1)
    print("全部通过 ✔")
