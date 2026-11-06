#!/usr/bin/env python3
"""Test the 555M model at each step of the dashboard, pausing the run.

    scripts/run555m_checks.py           # follow the run, in tmux session checks555m
    scripts/run555m_checks.py --now STEP  # test the last save, the run being stopped

At the first save past a step's tokens: stop the training, run the test of
that step and again those of all the steps before it, in order, into
logs/run555m/tests/DATE-STEP/KIND.txt, and start the training from that save.
The dashboard shows the last level and keeps all of them in its History tab.
The first chat try fine-tunes a copy first. The last step waits for the chat
model of the run. A failed test writes its error in place of its output, so it
is not retried at every save, and the training restarts anyway.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dashboard import MILESTONES, PROGRESS, ROOT, TESTS, levels, step_slug

LOGS = ROOT / "logs/run555m"
ANNI = ROOT / "build/555m/anni"
MODEL = ROOT / "models/run555m.bin"
CHAT = ROOT / "models/run555m-chat.bin"
FIRST_CHAT = ROOT / "models/run555m-first-chat-try.bin"
VOCAB = "data/mix100k.vocab"
WINDOW = 256
CHAT_STEP = "First chat try"
SEPARATOR = "\n\n---\n\n"
STEP_TOKENS = 131072
TOTAL_TOKENS = 15_590_156_032
STARTED = re.compile(rb"\[([\d-]+ [\d:]+)\] epoch \d+/\d+ starting")
SAVED = re.compile(rb"\[([\d-]+ [\d:]+)\] saved checkpoint \S+ \(epoch \d+, window (\d+)/")
WORD_PROMPTS = [
    "The history of the Roman Empire",
    "The main reason why the sky is blue is",
    "La ville de Paris est connue pour",
    "def fibonacci(n):\n",
    "#include <stdio.h>\n\nint main(void)\n{\n",
]
TEXT_PROMPTS = [
    "Once upon a time, in a small village by the sea,",
    "Climate change is a serious problem because",
    "Les enfants sont rentrés de l'école et",
    "To install the package, first open a terminal and",
    "class Stack:\n    def __init__(self):\n",
]
CHAT_PROMPTS = [
    "Hello, who are you?",
    "What is the capital of France?",
    "Write a Python function that reverses a string.",
    "Explique en une phrase pourquoi le ciel est bleu.",
]
FINAL_PROMPTS = [
    "Explain what a neural network is in two sentences.",
    "Give me a simple recipe for pancakes.",
    "Translate 'good morning, how are you?' into French.",
    "Write a C function that returns the length of a string.",
]
KINDS = dict(zip([m[0] for m in MILESTONES],
                 [WORD_PROMPTS, TEXT_PROMPTS, CHAT_PROMPTS, FINAL_PROMPTS]))
FINAL_STEP = MILESTONES[-1][0]


def say(text: str) -> None:
    print(f"[{time.strftime('%F %T')}] {text}", flush=True)


def alive(session: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", f"={session}"],
                          capture_output=True).returncode == 0


def trainer() -> int | None:
    out = subprocess.run(["pgrep", "-x", "anni"], capture_output=True, text=True).stdout
    for pid in out.split():
        args = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
        if b"lm train" in args and str(MODEL.relative_to(ROOT)).encode() in args:
            return int(pid)
    return None


def stop_run() -> None:
    pid = trainer()
    if pid:
        os.kill(pid, signal.SIGTERM)
    while alive("run555m"):
        time.sleep(1)


def start_run() -> None:
    subprocess.run(["scripts/run555m.sh"], cwd=ROOT, check=True)
    while alive("shots555m"):	# it leaves on its own once the run has gone
        if time.time() - stopped_at > 60:
            break
        time.sleep(1)
    subprocess.run(["scripts/dashboard_shots.sh"], cwd=ROOT, check=True)


def pair(prompt: str, answer: str) -> str:
    return f"PROMPT\n{prompt.rstrip()}\n\nANSWER\n{answer.rstrip()}"


def generate(model: Path, prompts: list[str]) -> str:
    out = []
    for prompt in prompts:
        done = subprocess.run(
            [str(ANNI), "lm", "generate", "--model", str(model), "--vocab", VOCAB,
             "--prompt", prompt, "--tokens", "120", "--temperature", "0.8",
             "--top-k", "40"],
            cwd=ROOT, capture_output=True, text=True, errors="replace",
            timeout=600, check=True)
        text = done.stdout.strip("\n")
        out.append(pair(prompt, text.removeprefix(prompt.rstrip()).lstrip(" ").removeprefix("\n")))
    return SEPARATOR.join(out)


def chat(model: Path, prompts: list[str]) -> str:
    done = subprocess.run(
        [str(ANNI), "chat", "--model", str(model), "--vocab", VOCAB, "--tokens", "120",
         "--temperature", "0.7", "--top-k", "40"],
        cwd=ROOT, input="\n".join(prompts) + "\n", capture_output=True, text=True,
        errors="replace", timeout=600, check=True, env=dict(os.environ, DISPLAY=""))
    answers = done.stdout.split("Assistant: ")[1:]
    answers = [a.removesuffix("User: ").strip() for a in answers]
    return SEPARATOR.join(pair(p, a) for p, a in zip(prompts, answers))


def finetune() -> None:
    """The fine-tune of scripts/run555m.sh, on a separate output."""
    with open(LOGS / "checks-finetune.log", "ab") as log:
        subprocess.run(
            [str(ANNI), "lm", "train", "--resume", str(MODEL), "--out", str(FIRST_CHAT),
             "--force", "--data", "data/chat100k_train_mids.bin",
             "--val-data", "data/chat100k_val_mids.bin", "--tokenizer", "bpe",
             "--vocab", VOCAB, "--window", str(WINDOW), "--stride", str(WINDOW),
             "--batch", "2", "--grad-accum", "64", "--epochs", "2", "--lr", "5e-5",
             "--lr-decay", "none", "--grad-clip", "1.0", "--weight-decay", "0.1",
             "--flash-attn", "--bf16", "--eval-every", "1", "--eval-windows", "256",
             "--seed", "2", "-d"],
            cwd=ROOT, stdout=log, stderr=log, check=True)


def stamp(raw: bytes) -> float:
    return time.mktime(time.strptime(raw.decode(), "%Y-%m-%d %H:%M:%S"))


def summary(data: bytes) -> str:
    """One line on the last save. Training time leaves out the pauses and the
    work a restart threw away, so it only runs from each start to its last save."""
    events = sorted([(m.start(), m) for m in STARTED.finditer(data)] +
                    [(m.start(), m) for m in SAVED.finditer(data)], key=lambda e: e[0])
    trained, start, last, save = 0.0, 0.0, 0.0, None
    for _, m in events:
        if m.re is STARTED:
            trained += last - start
            start = last = stamp(m[1])
        else:
            last, save = stamp(m[1]), m
    trained += last - start
    if not save:
        return "no save yet"
    tokens = int(save[2]) * WINDOW
    loss = PROGRESS.findall(data[max(0, save.start() - 4000):save.start()])
    parts = [f"{tokens / 1e9:.3f} G tokens ({tokens / TOTAL_TOKENS:.1%})",
             f"step {tokens // STEP_TOKENS:,}",
             f"{int(trained // 3600)} h {int(trained % 3600 // 60)} min of training"]
    if loss:
        parts.append(f"loss {float(loss[-1][2]):.2f}")
    parts.append(f"saved {time.strftime('%-d %b %H:%M', time.localtime(stamp(save[1])))}")
    return ", ".join(parts)


def run_level(level: str) -> None:
    """Test `level` and every kind before it, then publish them as one folder."""
    say(f"level {level}")
    failed = None
    if level == CHAT_STEP:
        try:
            finetune()
        except (OSError, subprocess.SubprocessError) as e:
            failed = e
    title = summary((LOGS / "pretrain.log").read_bytes())
    part = TESTS / f".part-{step_slug(level)}"
    shutil.rmtree(part, ignore_errors=True)
    part.mkdir(parents=True)
    names = [m[0] for m in MILESTONES]
    for kind in names[:names.index(level) + 1]:
        tuned = kind in (CHAT_STEP, FINAL_STEP)
        say(f"test {kind}")
        try:
            if tuned and failed:
                raise failed
            if tuned:
                body = chat(CHAT if level == FINAL_STEP else FIRST_CHAT, KINDS[kind])
            else:
                body = generate(MODEL, KINDS[kind])
        except (OSError, subprocess.SubprocessError) as e:
            body = f"test failed: {e}"
        line = title + (", plus chat fine-tune" if tuned else "")
        (part / f"{step_slug(kind)}.txt").write_text(
            f"{line}, tested {time.strftime('%-d %b %H:%M')}\n\n{body}\n", encoding="utf-8")
    part.rename(TESTS / f"{time.strftime('%Y%m%d-%H%M')}-{step_slug(level)}")
    say(f"level {level}: done")


def saves(data: bytes) -> tuple[str, int] | None:
    found = SAVED.findall(data)
    if not found:
        return None
    tokens = int(found[-1][1]) * WINDOW
    return f"save of {found[-1][0].decode()}, {tokens / 1e9:.3f} G tokens", tokens


def done(level: str) -> bool:
    return any(d.name.endswith("-" + step_slug(level)) for d in levels())


def follow() -> None:
    global stopped_at
    log = LOGS / "pretrain.log"
    pos = log.stat().st_size
    while not done(FINAL_STEP):
        with open(log, "rb") as f:
            f.seek(pos)
            data = f.read()
            pos = f.tell()
        save = saves(data)
        reached = [m[0] for m in MILESTONES if m[2] and save and save[1] >= m[2]]
        if reached and not done(reached[-1]):
            stopped_at = time.time()
            say(f"stopping the run after the {save[0]}")
            stop_run()
            try:
                run_level(reached[-1])
            finally:
                start_run()
                say("run restarted")
        if (LOGS / ".chat").exists():
            run_level(FINAL_STEP)
        time.sleep(20)
    say("all tests done")


stopped_at = 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--now", metavar="STEP", help="test the last save as this step")
    args = ap.parse_args()
    if args.now:
        if trainer():
            sys.exit("the run is training: stop it first")
        if args.now not in [m[0] for m in MILESTONES]:
            sys.exit(f"steps: {', '.join(m[0] for m in MILESTONES)}")
        run_level(args.now)
        return 0
    follow()
    return 0


if __name__ == "__main__":
    sys.exit(main())
