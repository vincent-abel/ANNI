#!/usr/bin/env python3
"""Serve the visual trainer and drive gui/ann-gui.

    python3 gui/server.py
"""

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUI = os.path.join(ROOT, "gui")
BIN = os.path.join(GUI, "ann-gui")

FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}

lock = threading.Lock()
proc = None


def num(value):
    return format(float(value), ".12g")


def build():
    subprocess.check_call(["make", "gui"], cwd=ROOT)


def drain(pipe):
    try:
        for line in pipe:
            sys.stderr.write(line)
    except Exception:
        pass


def spawn():
    child = subprocess.Popen(
        [BIN],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    threading.Thread(target=drain, args=(child.stderr,), daemon=True).start()
    return child


def stop_proc():
    global proc
    child = proc
    proc = None
    if child and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=1)
        except subprocess.TimeoutExpired:
            child.kill()


def rpc(line):
    global proc
    last = "the C trainer stopped"
    with lock:
        for _attempt in (1, 2):
            if proc is None or proc.poll() is not None:
                proc = spawn()
            try:
                proc.stdin.write(line + "\n")
                proc.stdin.flush()
                resp = proc.stdout.readline()
            except (BrokenPipeError, OSError) as exc:
                last = str(exc)
                stop_proc()
                continue
            if not resp:
                last = "the C trainer closed its output"
                stop_proc()
                continue
            return json.loads(resp)
    raise RuntimeError(last)


def ints(values, name):
    out = []
    for value in values:
        n = int(value)
        if n < 1:
            raise ValueError(name + " must be positive")
        out.append(n)
    return out


def command_for(path, msg):
    if path == "/api/state":
        return "state"
    if path == "/api/create":
        dataset = msg.get("dataset")
        if dataset not in ("xor", "custom"):
            raise ValueError("dataset must be xor or custom")
        seed = int(msg["seed"])
        learn = float(msg["lr"])
        if learn <= 0:
            raise ValueError("learning rate must be positive")
        n_inputs = int(msg["inputs"])
        sizes = ints(msg.get("sizes") or [], "layer size")
        if not sizes:
            raise ValueError("provide at least one layer size")
        return "create %s %d %s %d %s" % (
            dataset, seed, num(learn), n_inputs, " ".join(str(s) for s in sizes)
        )
    if path == "/api/train":
        steps = int(msg.get("steps", 1))
        if steps < 1:
            raise ValueError("steps must be a positive integer")
        if "lr" in msg and msg["lr"] is not None:
            return "train %d %s" % (steps, num(msg["lr"]))
        return "train %d" % steps
    if path == "/api/sample":
        return "sample %d" % int(msg["index"])
    if path == "/api/values":
        inputs = [float(v) for v in msg.get("inputs") or []]
        targets = [float(v) for v in msg.get("targets") or []]
        return "values %d %s %d %s" % (
            len(inputs),
            " ".join(num(v) for v in inputs),
            len(targets),
            " ".join(num(v) for v in targets),
        )
    raise ValueError("unknown request")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s\n" % (fmt % args))

    def send_json(self, obj, code=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_file(self, name, content_type):
        path = os.path.join(GUI, name)
        with open(path, "rb") as handle:
            data = handle.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def read_body(self):
        n = int(self.headers.get("Content-Length", "0") or "0")
        if n < 0 or n > 2000000:
            raise ValueError("request is too large")
        if n == 0:
            return {}
        raw = self.rfile.read(n)
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        return data

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/state":
            try:
                self.send_json(rpc("state"))
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)})
            return
        item = FILES.get(path)
        if not item:
            self.send_error(404)
            return
        self.send_file(item[0], item[1])

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            msg = self.read_body()
            line = command_for(path, msg)
            self.send_json(rpc(line))
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)})


class Server(ThreadingHTTPServer):
    allow_reuse_address = True


def main():
    build()
    for port in range(8765, 8780):
        try:
            httpd = Server(("127.0.0.1", port), Handler)
        except OSError:
            continue
        print("ANN visual trainer  http://127.0.0.1:%d" % port, flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("")
        finally:
            stop_proc()
            httpd.server_close()
        return
    raise SystemExit("no free port between 8765 and 8779")


if __name__ == "__main__":
    main()
