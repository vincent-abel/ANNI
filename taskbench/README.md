# Small coding tasks

The task set of a small coding benchmark,
copied unchanged: `tasks.js` (20 tasks, each with its setup, request and
check) and `project/` (a dependency-free module whose tests run with `node test.js`). The
`fake` and `fakeExpect` fields of the tasks only serve that benchmark's own runner and are not used here.

```sh
node taskbench/run.js --model models/talk170m-2k-chat.bin          # all tasks
node taskbench/run.js --model M --tasks fix-clamp,explain-retry     # a subset
node taskbench/run.js --selftest                                    # grading check, no model
```

`run.js` replaces the agent harness: AnnI has no tools, so each task is a single turn.
The prompt, in the chat fine-tune's `User:` / `Assistant:` format, holds the request, the
files it names (its attachments, or every `src/` file when it names none, cut to fit the
context) and the reply format: `File: <path>` then the complete new content in a fenced
block. Those blocks are written into a fresh copy of `project/`, then the task's own check
runs on it: tests, hidden assertions, file contents, or the reply text for the question.
Checks on tool use (`find-failing-test` wants the tests run) cannot pass.

Generation is greedy by default (`--temperature`), with a 512-token reply (`--tokens`) in
a 2048-token context (`--ctx`). anni runs with `ANN_GPU_ONLY=1`, so a missing GPU fails
the task instead of generating on the CPU; prompts past 256 tokens are prefilled by the
device decoder. Results go to `logs/taskbench/<time>.json` with the prompt files, the reply
and every check.
