# PPO-Lagrangian Baseline v1

This protocol defines the constrained learned baseline used between standard PPO and RCWA-RL.

## Scientific role

PPO-Lagrangian v1 is intentionally **expected-constraint aware but not tail-risk aware**. It shares the PPO v1 actor, observation, action distribution, training weather/eta scheduler, interaction budget, validation protocol and checkpoint-selection contract.

For weather year `w` and requested protection `eta`, define retention

\[
R_\pi(w)=Y_\pi(w)/Y_{ref}(w).
\]

The water reward is

\[
r_t=-I_t/495.
\]

The signed terminal constraint cost is

\[
C_\eta(w)=\eta-R_\pi(w).
\]

For every eta separately, PPO-Lagrangian trains against

\[
\boxed{E_w[C_\eta(w)\mid\eta]\le 0}
\]

or equivalently

\[
\boxed{E_w[R_\pi(w)\mid\eta]\ge\eta}.
\]

The cost is deliberately **not** clipped at zero. A positive-yield-surplus weather year can compensate a deficit year under this expected constraint. That is the scientific distinction from RCWA-RL, whose formal constraint is lower-tail LCVaR.

## Lagrangian update

There are three independent multipliers:

\[
\lambda_{0.90},\lambda_{0.95},\lambda_{0.98}\ge0.
\]

All initialize to 1.0. For the 18 same-eta training episodes in update `k`,

\[
\lambda_{\eta,k+1}=
\max\left(0,\lambda_{\eta,k}+0.05\,\overline C_{\eta,k}\right).
\]

The actor and both critics consume the whole balanced on-policy rollout using the frozen `lambda_k`; only after that primal update is complete is the projected dual update applied. No extra DSSAT trajectory is used for the dual update.

The `1.0` initial multiplier and `0.05` dual learning-rate scale are frozen before formal training and match the penalty initialization / learning-rate scale used in OpenAI Safety Starter Agents' PPO-Lagrangian implementation.

## Policy and critics

The actor is identical to PPO v1:

```text
79D -> 256 -> 128
             |- Bernoulli irrigation gate
             `- active-only state-dependent tanh-Gaussian amount
```

PPO-Lagrangian adds a separate cost critic but does not enlarge or otherwise improve the actor:

- reward critic: 79 -> 256 -> 128 -> scalar;
- cost critic: 79 -> 256 -> 128 -> scalar.

Both use `gamma=1` and `GAE lambda=1` because the relevant crop/yield consequence is terminal.

The policy advantage is

\[
A^L_t=A^r_t-\lambda_{\eta(t)}A^c_t.
\]

The eta-specific multipliers are applied first; the resulting combined advantage is normalized once across the complete 6750-transition batch and then used in the same clipped PPO objective as PPO v1.

## Fixed optimizer settings

- learning rate: `1e-4`;
- Adam betas: `(0.9, 0.999)`;
- Adam epsilon: `1e-8`;
- PPO clip: `0.2`;
- epochs: `10`;
- minibatch: `450`;
- reward-value coefficient: `0.5`;
- cost-value coefficient: `0.5`;
- entropy coefficient: `0`;
- max global gradient norm: `0.5`.

## Interaction fairness

Every update uses exactly:

\[
18\text{ years}\times3\eta=54\text{ complete episodes}
\]

and

\[
54\times125=6750\text{ transitions}.
\]

Per seed the maximum training budget is exactly the same as PPO v1:

\[
10,800\text{ episodes}=1,350,000\text{ decisions}.
\]

The dual variables and cost critic are trained only from these same trajectories. They do not receive additional environment samples.

Seeds remain `11, 21, 31, 41, 51`.

## Validation and checkpoint selection

Recovery checkpoints are written every update. Scientific candidate checkpoints and deterministic validation occur only at updates `10,20,...,200`.

The same frozen 15-cell validation and learned-method checkpoint selector used by PPO v1 are reused without modification. Model selection and final reporting still use the formal Risk Contract v1 LCVaR metric. Therefore expected-constraint training does not get a different or easier selection criterion.

2023-2025 station weather remains locked until all learned methods and selected checkpoints are frozen.

## Comparison interpretation

The intended comparison is:

```text
PPO               : scalar expected-return baseline, no explicit constraint optimizer
PPO-Lagrangian    : expected-retention constraint, no tail-risk objective
RCWA-RL           : formal lower-tail LCVaR retention constraint
```

Any change to this protocol after formal PPO-Lagrangian training begins requires a new protocol version.
