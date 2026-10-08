# Adam optimizer

## What problem it solves

Plain gradient descent uses the same learning rate for every weight. Some weights need big steps, others tiny ones, and noisy gradients can jitter the path. **Adam** keeps a short memory of past gradients (momentum) and of past squared gradients (adaptive scale) so each parameter gets its own effective step size.

## How it works (layman)

For every weight, Adam tracks two running averages: “which way have we been going?” (\(m\)) and “how jumpy is the gradient?” (\(v\)). The update follows the smoothed direction but divides by how jumpy that weight has been, so calm weights move more freely and wild ones take smaller steps. A bias correction boosts \(m\) and \(v\) early on when the averages are still cold-starting from zero.

## The math

With gradient \(g_t\), defaults \(\beta_1=0.9\), \(\beta_2=0.999\), \(\varepsilon=10^{-8}\):

\[
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,
\quad
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
\]

\[
\hat{m}_t = \frac{m_t}{1-\beta_1^t},
\quad
\hat{v}_t = \frac{v_t}{1-\beta_2^t},
\quad
\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\varepsilon}
\]

## How it shows up in this repo

- [`optim.h`](../optim.h) / [`optim.c`](../optim.c): `adam_create` walks the network and builds slots pointing at each bias/weight and its `dbias`/`dw`.
- `adam_step` applies the update and clears grads; `network_train_adam` runs forward/backward/Adam per sample.
- Tests train XOR with Adam instead of `network_update`.

## Tiny example

A weight that always gets gradient \(\approx 0.01\) builds a small \(v\) and can take larger relative steps than one slammed by gradients of size \(10\).
