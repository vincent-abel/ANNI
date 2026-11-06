#!/usr/bin/env python3
"""Test the 555M model at each step of the dashboard, pausing the run.

    scripts/run555m_checks.py           # follow the run, in tmux session checks555m
    scripts/run555m_checks.py --now STEP  # test the last save, the run being stopped

At the first save past a step's tokens: stop the training, sample the model on
the GPU into logs/run555m/tests/STEP.txt (the dashboard shows those at the
bottom), and start the training again from that save. The first chat try
fine-tunes a copy first. The last step waits for the chat model of the run.
A failed test writes its error to the same file, so it is not retried at every
save, and the training restarts anyway.
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dashboard import MILESTONES, ROOT, TESTS, step_slug

LOGS = ROOT / "logs/run555m"
ANNI = ROOT / "build/555m/anni"
MODEL = ROOT / "models/run555m.bin"
CHAT = ROOT / "models/run555m-chat.bin"
FIRST_CHAT = ROOT / "models/run555m-first-chat-try.bin"
VOCAB = "data/mix100k.vocab"
WINDOW = 256
CHAT_STEP = "First chat try"
SEPARATOR = "\n\n---\n\n"
SAVED = re.compile(rb"\[([\d-]+ [\d:]+)\] saved checkpoint \S+ \(epoch \d+, window (\d+)/")
PROMPTS = [
    "The history of the Roman Empire",
    "The main reason why the sky is blue is",
    "La ville de Paris est connue pour",
    "def fibonacci(n):\n",
    "#include <stdio.h>\n\nint main(void)\n{\n",
]
CHAT_PROMPTS = [
    "Hello, who are you?",
    "What is the capital of France?",
    "Write a Python function that reverses a string.",
    "Explique en une phrase pourquoi le ciel est bleu.",
]


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


def generate(model: Path) -> str:
    out = []
    for prompt in PROMPTS:
        done = subprocess.run(
            [str(ANNI), "lm", "generate", "--model", str(model), "--vocab", VOCAB,
             "--prompt", prompt, "--tokens", "120", "--temperature", "0.8",
             "--top-k", "40"],
            cwd=ROOT, capture_output=True, text=True, errors="replace",
            timeout=600, check=True)
        text = done.stdout.strip("\n")
        out.append(pair(prompt, text.removeprefix(prompt.rstrip()).lstrip(" ").removeprefix("\n")))
    return SEPARATOR.join(out)


def chat(model: Path) -> str:
    done = subprocess.run(
        [str(ANNI), "chat", "--model", str(model), "--vocab", VOCAB, "--tokens", "120",
         "--temperature", "0.7", "--top-k", "40"],
        cwd=ROOT, input="\n".join(CHAT_PROMPTS) + "\n", capture_output=True, text=True,
        errors="replace", timeout=600, check=True, env=dict(os.environ, DISPLAY=""))
    answers = done.stdout.split("Assistant: ")[1:]
    answers = [a.removesuffix("User: ").strip() for a in answers]
    return SEPARATOR.join(pair(p, a) for p, a in zip(CHAT_PROMPTS, answers))


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


def test(name: str, saved: str) -> None:
    say(f"test {name}: {saved}")
    try:
        if name == CHAT_STEP:
            finetune()
            body = chat(FIRST_CHAT)
        elif MILESTONES[-1][0] == name:
            body = chat(CHAT)
        else:
            body = generate(MODEL)
    except (OSError, subprocess.SubprocessError) as e:
        body = f"test failed: {e}"
    TESTS.mkdir(parents=True, exist_ok=True)
    path = TESTS / f"{step_slug(name)}.txt"
    part = path.with_suffix(".part")
    part.write_text(f"{saved}, tested {time.strftime('%F %H:%M')}\n\n{body}\n", encoding="utf-8")
    part.rename(path)
    say(f"test {name}: done")


def saves(data: bytes) -> tuple[str, int] | None:
    found = SAVED.findall(data)
    if not found:
        return None
    tokens = int(found[-1][1]) * WINDOW
    return f"save of {found[-1][0].decode()}, {tokens / 1e9:.3f} G tokens", tokens


def done(name: str) -> bool:
    return (TESTS / f"{step_slug(name)}.txt").exists()


def follow() -> None:
    global stopped_at
    log = LOGS / "pretrain.log"
    pos = log.stat().st_size
    while not all(done(m[0]) for m in MILESTONES):
        with open(log, "rb") as f:
            f.seek(pos)
            data = f.read()
            pos = f.tell()
        save = saves(data)
        due = [m[0] for m in MILESTONES if m[2] and save and save[1] >= m[2] and not done(m[0])]
        if due:
            stopped_at = time.time()
            say(f"stopping the run after the {save[0]}")
            stop_run()
            try:
                for name in due:
                    test(name, save[0])
            finally:
                start_run()
                say("run restarted")
        if (LOGS / ".chat").exists() and not done(MILESTONES[-1][0]):
            test(MILESTONES[-1][0], "chat model")
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
        save = saves((LOGS / "pretrain.log").read_bytes())
        test(args.now, save[0] if save else "last save")
        return 0
    follow()
    return 0


if __name__ == "__main__":
    sys.exit(main())
