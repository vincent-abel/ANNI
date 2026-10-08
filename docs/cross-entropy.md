# Cross-entropy

## What problem it solves

After softmax gives class probabilities, training needs a **loss**: how wrong was the predicted distribution vs the true label? Cross-entropy is large when the true class got a tiny probability and small when that probability is near 1.

## How it works (layman)

If the model says “30% chance this is class 2” and the answer *is* class 2, the loss is \(-\log(0.3)\). Confident and correct → small loss; confident and wrong → huge loss. You can also compute it straight from logits with a stable formula that never materializes an intermediate probability vector (same answer, fewer overflow risks).

## The math

For probabilities \(\mathbf{p}\) and true class \(y\):

\[
\operatorname{CE}(\mathbf{p}, y) = -\log p_y
\]

From logits \(\mathbf{z}\) (stable log-softmax form):

\[
\operatorname{SCE}(\mathbf{z}, y) = -(z_y - \max_j z_j - \log\sum_k e^{z_k - \max_j z_j})
\]

## How it shows up in this repo

- [`ops.c`](../num/ops.c): `vec_cross_entropy(probs, class_index)` clamps tiny \(p\) then returns \(-\log p\).
- `vec_softmax_cross_entropy(logits, class_index)` uses the stable combined formula above.
- The existing MLP still trains with MSE on XOR; these ops are the Numerical Engine loss for multi-class work ahead.

## Tiny example

Probabilities `[0.1, 0.7, 0.2]`, true class `1` → CE \(= -\log(0.7) \approx 0.357\).
