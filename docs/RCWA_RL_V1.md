# RCWA-RL v1

RCWA-RL is the proposed tail-risk-aware learned irrigation method for the first AWM study. It is frozen against `awm-risk-contract-v1` before formal RCWA-RL training.

## Objective and constraint

For requested protection level `eta in {0.90,0.95,0.98}`:

\[
\max_\pi\;\mathbb E[-I_{policy}/495]
\]

subject to

\[
\mathrm{LCVaR}_{0.20}(R_\pi\mid\eta)\ge \eta,
\qquad
R_\pi(w)=Y_\pi(w)/Y_{ref}(w).
\]

The actor is the same 79D conditional hierarchical irrigation policy used by PPO v1. The same weather scheduler, water budget, observation normalization, five seeds, interaction budget, candidate cadence, deterministic validation and checkpoint selector are used.

## Batch tail estimator

Each update contains exactly 18 training weather years for each eta. Let their retentions be sorted as

\[
r_{(1)}\le\cdots\le r_{(18)}.
\]

With `alpha=0.20`, `alpha*n=3.6`. RCWA-RL sets

\[
\tau_\eta=r_{(4)},
\]

the empirical lower-20% quantile. It then computes

\[
\widehat L_\eta
=
\tau_\eta-rac1\alpha\frac1{18}
\sum_i(\tau_\eta-r_i)_+.
\]

The implementation requires this value to equal the exact fractional order-statistic estimator in Risk Contract v1. Therefore the scientific constraint metric is not replaced by an approximate definition during training.

## Actor tail-risk signal

For an episode with retention `R`, the terminal risk signal is

\[
h_\eta(R)=\frac1\alpha(\tau_\eta-R)_+.
\]

`tau_eta` is frozen for the entire PPO optimization step. The term `eta-tau_eta` is omitted from the actor/risk-critic return because, at the batch optimum, it is action-distribution independent under the Rockafellar-Uryasev envelope argument. The full constraint violation used by the dual is still

\[
g_\eta=\eta-\widehat L_\eta.
\]

Thus the actor is updated with

\[
A^{actor}=A^{water}-\lambda_\eta A^{tail},
\]

while the dual update uses the exact empirical LCVaR violation.

## Eta-specific duals

Three independent multipliers are maintained:

\[
\lambda_{0.90},\lambda_{0.95},\lambda_{0.98}\ge0.
\]

They initialize at `1.0` and are updated after the primal PPO step:

\[
\lambda_{\eta,k+1}
=
\max(0,\lambda_{\eta,k}+0.05\,g_{\eta,k}).
\]

The same rollout supplies actor learning, risk-critic learning, empirical CVaR estimation and dual updates. RCWA-RL receives zero additional DSSAT trajectories.

## Fairness controls

For the same seed, RCWA-RL must start with exactly the same actor and reward-critic parameters as PPO v1. Extra model capacity is limited to one risk critic and three scalar dual variables. PPO optimizer settings remain unchanged: learning rate `1e-4`, `gamma=1`, GAE lambda `1`, clip `0.2`, ten epochs, minibatch `450`, no entropy bonus and gradient clipping `0.5`.

Each seed receives at most 200 updates = 10,800 training episodes = 1.35M decisions. Scientific candidates remain updates 10,20,...,200. Validation remains deterministic over 2018-2022 × three eta values. Station years 2023-2025 remain locked until final checkpoint freeze.

## Distinction from comparators

- PPO v1: scalar expected-return baseline, no explicit distributional constraint.
- PPO-Lagrangian v1: eta-specific expected-retention constraint `E[R|eta]>=eta`.
- RCWA-RL v1: eta-specific lower-tail constraint `LCVaR_0.20(R|eta)>=eta`.

No expected-retention surrogate may replace the RCWA-RL dual violation, and no eta groups may be pooled.
