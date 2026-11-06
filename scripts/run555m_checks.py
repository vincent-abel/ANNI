#!/usr/bin/env python3
"""Test the 555M model at each step of the dashboard, pausing the run.

    scripts/run555m_checks.py           # follow the run, in tmux session checks555m
    scripts/run555m_checks.py --now STEP  # test the last save, the run being stopped

At the first save past a step's tokens: stop the training, run the test of
that step and again those of all the steps before it, in order, into
logs/run555m/tests/DATE-STEP/KIND.txt, and start the training from that save.
The dashboard shows the last level and keeps all of them in its History tab.
Each step first keeps the weights of that save, for comparisons, as
models/run555m-base-STEP.bin. From the first chat try on, it also fine-tunes its
own copy, the chat kinds run on it, and only its weights are kept. Each step
also scores the pretrained model on HellaSwag, into the dashboard tab of that
name; scripts/hellaswag_build.sh must have been run. The last step
waits for the chat model of the run. A failed test writes its error in place of its output, so it
is not retried at every save, and the training restarts anyway.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dashboard import HELLASWAG, MILESTONES, PROGRESS, ROOT, TESTS, hellaswag_rows, levels, step_slug

LOGS = ROOT / "logs/run555m"
ANNI = ROOT / "build/555m/anni"
BENCH = ROOT / "build/hellaswag/anni"	# the same model code, with the bench
MODEL = ROOT / "models/run555m.bin"
CHAT = ROOT / "models/run555m-chat.bin"
VOCAB = "data/mix100k.vocab"
WINDOW = 256
CHAT_STEP = "First chat try"
SEPARATOR = "\n\n---\n\n"
EARLIER_WEIGHTS = {"Readable text": ("models/run555m-base-0.653g.bin", 0.653)}	# before the bench
STEP_TOKENS = 131072
TOTAL_TOKENS = 15_590_156_032
PARAMS = 555_762_080
ADAM_HEADER = 20	# has_adam, epoch, step, t, n as int32
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
SIMPLE_PROMPTS = [
    "How many days are there in a week?",
    "What do bees make?",
    "Name three colors.",
    "Quelle est la couleur du ciel ?",
]
STEADY_PROMPTS = [
    "Give me three tips to sleep better.",
    "Write a short email asking for a day off.",
    "Why do leaves fall in autumn?",
    "Écris un petit poème sur la mer.",
]
USEFUL_PROMPTS = [
    "What is the difference between a list and a tuple in Python?",
    "If I have 3 apples and eat one, how many are left?",
    "Summarize the story of Cinderella in three sentences.",
    "Write a C function that adds two integers.",
]
FINAL_PROMPTS = [
    "Explain what a neural network is in two sentences.",
    "Give me a simple recipe for pancakes.",
    "Translate 'good morning, how are you?' into French.",
    "Write a C function that returns the length of a string.",
]
KINDS = dict(zip([m[0] for m in MILESTONES],
                 [WORD_PROMPTS, TEXT_PROMPTS, CHAT_PROMPTS, SIMPLE_PROMPTS,
                  STEADY_PROMPTS, USEFUL_PROMPTS, FINAL_PROMPTS], strict=True))
FINAL_STEP = MILESTONES[-1][0]
NAMES = [m[0] for m in MILESTONES]
TUNED = NAMES[NAMES.index(CHAT_STEP):]	# kinds that need a chat model


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


def weights_only(src: Path, dst: Path) -> None:
    """Write checkpoint `src` to `dst` without its Adam state, m and v per
    parameter: two thirds of the file. `dst` can be `src`."""
    end = src.stat().st_size - ADAM_HEADER - 8 * PARAMS
    tmp = dst.with_name(dst.name + ".tmp")
    with open(src, "rb") as f:
        f.seek(end)
        head = struct.unpack("<5i", f.read(ADAM_HEADER))
        if head[0] != 1 or head[4] != PARAMS:
            raise ValueError(f"{src.name}: no Adam section at byte {end}")
        f.seek(0)
        with open(tmp, "wb") as out:
            left = end
            while left:
                left -= out.write(f.read(min(left, 1 << 24)))
            out.write(struct.pack("<i", 0))
    tmp.replace(dst)


def keep_weights(src: Path, dst: Path) -> None:
    """A failure here only costs disk space or a comparison point."""
    try:
        weights_only(src, dst)
        say(f"kept the weights in {dst.name}")
    except (OSError, ValueError, struct.error) as e:
        say(f"weights of {src.name} not cut: {e}")


def tuned_copy(level: str) -> Path:
    return ROOT / f"models/run555m-{step_slug(level)}.bin"


def finetune(level: str) -> None:
    """The fine-tune of scripts/run555m.sh, on a separate output."""
    with open(LOGS / "checks-finetune.log", "ab") as log:
        subprocess.run(
            [str(ANNI), "lm", "train", "--resume", str(MODEL), "--out", str(tuned_copy(level)),
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


def save_title(giga: float) -> str | None:
    """The summary line of the save at `giga` billion tokens."""
    data = (LOGS / "pretrain.log").read_bytes()
    for m in SAVED.finditer(data):
        if round(int(m[2]) * WINDOW / 1e9, 3) == giga:
            return summary(data[:m.end()])
    return None


def hellaswag(step: str, model: Path, title: str) -> None:
    """Add a row for `step`, an error row if it fails: the training must go on."""
    say(f"hellaswag on {model.name}, as {step}")
    tokens = re.search(r"([\d.]+) G tokens", title)
    loss = re.search(r"loss ([\d.]+)", title)
    row = {"step": step, "tested": time.strftime("%-d %b %H:%M"),
           "tokens": float(tokens[1]) if tokens else None,
           "nll": float(loss[1]) if loss else None}
    try:
        done = subprocess.run(
            [str(BENCH), "lm", "hellaswag", "--model", str(model), "--vocab", VOCAB],
            cwd=ROOT, capture_output=True, text=True, errors="replace",
            timeout=3600, check=True)
        row |= json.loads(done.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as e:
        row["error"] = str(e)[:200]
    with open(HELLASWAG, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    say(f"hellaswag: {row.get('acc_norm', row.get('error'))}")


def bench(level: str, title: str) -> None:
    """Score this level, and first the older weights that have no score yet."""
    tried = {row["step"] for row in hellaswag_rows()}
    for step, (path, giga) in EARLIER_WEIGHTS.items():
        earlier = save_title(giga)
        if step not in tried and earlier and (ROOT / path).exists():
            hellaswag(step, ROOT / path, earlier)
    hellaswag(level, MODEL, title)


def run_level(level: str) -> None:
    """Test `level` and every kind before it, then publish them as one folder."""
    say(f"level {level}")
    failed = None
    if level != FINAL_STEP:
        keep_weights(MODEL, ROOT / f"models/run555m-base-{step_slug(level)}.bin")
    if level in TUNED and level != FINAL_STEP:
        try:
            finetune(level)
        except (OSError, subprocess.SubprocessError) as e:
            failed = e
    title = summary((LOGS / "pretrain.log").read_bytes())
    part = TESTS / f".part-{step_slug(level)}"
    shutil.rmtree(part, ignore_errors=True)
    part.mkdir(parents=True)
    for kind in NAMES[:NAMES.index(level) + 1]:
        tuned = kind in TUNED
        say(f"test {kind}")
        try:
            if tuned and failed:
                raise failed
            if tuned:
                body = chat(CHAT if level == FINAL_STEP else tuned_copy(level), KINDS[kind])
            else:
                body = generate(MODEL, KINDS[kind])
        except (OSError, subprocess.SubprocessError) as e:
            body = f"test failed: {e}"
        line = title + (", plus chat fine-tune" if tuned else "")
        (part / f"{step_slug(kind)}.txt").write_text(
            f"{line}, tested {time.strftime('%-d %b %H:%M')}\n\n{body}\n", encoding="utf-8")
    part.rename(TESTS / f"{time.strftime('%Y%m%d-%H%M')}-{step_slug(level)}")
    bench(level, title)
    if level in TUNED and level != FINAL_STEP and not failed:
        copy = tuned_copy(level)
        keep_weights(copy, copy)
        copy.with_name(copy.name + ".best").unlink(missing_ok=True)
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
