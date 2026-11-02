# Question bench

A fixed set of 40 single-turn questions to ask every new model, so two models can be compared on the same answers. [`taskbench/`](../taskbench/README.md) is the hard test, multi-file project edits; this one checks the basics.

- **20 general questions** (capitals, counts, simple facts and arithmetic). An answer passes when it matches the expected word or number, for example `paris` or `19` (case-insensitive, whole words).
- **20 code questions:** 15 in Python and 5 in JavaScript, each asking for one small function such as `reverse_string(s)`, `fizzbuzz(n)` or `sumArray(arr)`. The first fenced block of the answer runs with the question's tests appended: `python3 -I` or `node`, in a temporary directory, with stdin closed and a 10 s limit. It passes when every assertion holds.

For code, the results also count two weaker signals, so a small model's progress shows before its code is right:

- **block:** the answer has a code block;
- **ran:** that code runs without a syntax error, even if a test fails.

```sh
node qbench/run.js --model models/talk170m-2k-chat.bin       # all 40 questions
node qbench/run.js --model M --only code                      # or --only general
node qbench/run.js --compare logs/qbench/A.json logs/qbench/B.json [C.json ...]
node qbench/run.js --selftest                                 # grading check, no model
```

Prompts use the chat fine-tune's format, `User: <question>\nAssistant:`.

- **Replies:** greedy (`--temperature 0`), at most 64 tokens for general questions and 256 for code (`--tokens` sets both). The reply stops at the next `User:` or `Assistant:` line.
- **Device:** anni generates on the GPU when it can, and on the CPU otherwise, for example while llama-server holds the GPU. `--gpu-only` fails instead. Greedy output can differ slightly between the two devices, so compare runs made on the same one.
- **Results** go to `logs/qbench/<model>-<time>.json`, with every reply, pass/fail, and the error line of a failed program. `--compare` prints a pass table of several result files side by side.

The answers' code runs on this machine. Questions and tests are in [`questions.js`](questions.js); keep them unchanged once a model has been scored, or older results stop being comparable.
