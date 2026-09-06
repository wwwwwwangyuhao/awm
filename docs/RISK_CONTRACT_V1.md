# AWM Risk Contract v1

This document freezes the yield-protection risk semantics used by the first AWM learned irrigation study. The machine-readable source of truth is `configs/risk_contract_v1.json`.

## 1. Freeze boundary

Risk Contract v1 is frozen **after** the agricultural-baseline development sweep and **before** any PPO or RCWA-RL training. No 2023-2025 Huaxing station final-test outcome has been used to choose the contract.

Changing the reference yield, lower-tail probability, protection levels, weather weighting or empirical LCVaR estimator after learned-policy training begins requires a new risk-contract version.

## 2. Same-weather yield reference

For weather year `w`, the reference is the yield of the frozen quota-normalized Huaxing conventional W100 schedule under exactly the same weather year and non-water management:

\[
Y_{\mathrm{ref}}(w)=Y_{\mathrm{Conventional,W100}}(w).
\]

The reference schedule uses 540 mm total seasonal irrigation: 45 mm fixed preplant establishment irrigation plus 495 mm postplant irrigation on the frozen Huaxing conventional timing pattern.

The development reference table for ERA5 2000-2022 is frozen in:

`configs/yield_reference_v1_development.json`.

No 2023-2025 station reference yield is stored there. Station references are generated only after learned methods, checkpoints, hyperparameters and this contract are frozen.

`Y_ref(w)` is an offline constraint/evaluation quantity. It is never exposed to the actor observation.

## 3. Risk random variable: yield retention

For a policy `pi` under weather year `w`, define

\[
R_\pi(w)=\frac{Y_\pi(w)}{Y_{\mathrm{ref}}(w)}.
\]

`R_pi` is **not clipped at 1**. If a policy produces more yield than the frozen W100 reference in a particular year, `R_pi(w)>1` is retained. Clipping would erase a real difference in outcomes and bias the lower-tail calculation upward or downward depending on the sample.

The same-year normalization prevents absolute high-yield and low-yield weather years from dominating the risk metric merely because their DSSAT yield scales differ.

## 4. Weather-year probability measure

Within each split, every weather year receives equal probability mass.

- Training distribution: ERA5 2000-2017, 18 equally weighted years.
- Validation distribution: ERA5 2018-2022, 5 equally weighted years.
- Locked final test: Huaxing station 2023-2025, 3 years, inaccessible during development.

Risk optimization gradients may use only training years. Validation years may later be used for checkpoint/hyperparameter selection under a separately frozen learned-method selection rule. Final-test station outcomes may not influence training, reward design, hyperparameters, checkpoint selection, eta, alpha or reference semantics.

## 5. Lower CVaR convention

The formal risk measure is lower conditional value at risk with

\[
\boxed{\alpha=0.20}.
\]

Here `alpha` means **lower-tail probability mass**. It does not mean an 80%, 95% or 99% confidence level.

For a retention random variable `R`,

\[
\mathrm{LCVaR}_{\alpha}(R)
=\frac{1}{\alpha}\int_0^\alpha Q_R(u)\,du,
\]

where `Q_R` is the lower quantile function. An equivalent reward-oriented optimization form is

\[
\mathrm{LCVaR}_{\alpha}(R)
=\sup_{\tau}\left[\tau-\frac{1}{\alpha}\mathbb{E}(\tau-R)_+\right].
\]

Thus higher LCVaR is better: it is the mean retention in the adverse lower tail.

## 6. Exact finite-sample estimator

The code does not delegate CVaR semantics to a library quantile interpolation setting.

For `n` equally weighted retention observations, sort

\[
r_{(1)}\le\cdots\le r_{(n)}.
\]

Let

\[
m=\alpha n,\qquad k=\lfloor m\rfloor,\qquad f=m-k.
\]

Then

\[
\widehat{\mathrm{LCVaR}}_{\alpha}
=
\frac{\sum_{i=1}^{k}r_{(i)}+f\,r_{(k+1)}}{m},
\]

with the fractional term omitted when `f=0`.

For protocol v1:

- training: `n=18`, so `alpha*n=3.6`; the estimator uses the three worst years plus 60% weight on the fourth-worst year;
- validation: `n=5`, so `alpha*n=1`; the estimator is exactly the worst validation year;
- final station set: `n=3`, so `alpha*n=0.6`; the empirical estimator equals the worst of the three years. Because three station years cannot precisely estimate a population tail, all three per-year outcomes must also be reported individually.

The contract makes an empirical historical-weather risk statement. It does not silently convert these finite weather samples into a population confidence interval.

## 7. Registered protection levels

The conditional policy receives the requested protection target as the existing observation feature `yield_target_fraction`.

Risk Contract v1 registers exactly three primary levels:

\[
\boxed{\eta\in\{0.90,\;0.95,\;0.98\}}.
\]

Interpretation:

- `eta=0.90`: standard yield protection;
- `eta=0.95`: strict yield protection;
- `eta=0.98`: near-full-water yield protection.

The target is constant within an episode. During conditional-policy training, the three registered targets must receive equal sampling weight. Every target is reported separately; no favorable eta may be selected after seeing learned-policy results.

The formal constraint at requested level `eta` is

\[
\boxed{
\mathrm{LCVaR}_{0.20}(R_\pi)\ge\eta
}.
\]

The risk margin is

\[
m_\pi(\eta)=\mathrm{LCVaR}_{0.20}(R_\pi)-\eta.
\]

A policy is feasible when `m_pi(eta)>=0`. Protocol v1 introduces no scientific feasibility slack. Machine epsilon used for floating-point comparisons is not a statistical tolerance.

## 8. Development-only sanity evidence before learned training

The 207-episode frozen agricultural-baseline sweep was used only to verify that the registered protection levels span meaningfully different regimes before learned training.

Using same-year Conventional-W100 normalization:

| Baseline | Split | LCVaR_0.20 retention |
|---|---|---:|
| Conventional W100 | train | 1.000 |
| Conventional W80 | train | 0.977 |
| REW W80 | train | 0.956 |
| Conventional W80 | validation | 0.929 |
| REW W80 | validation | 0.940 |

Thus `0.90`, `0.95`, and `0.98` are not three numerically redundant labels: they range from clearly attainable water-saving protection to a near-reference regime. These values were frozen before any learned-policy result existed and cannot be changed in v1 based on later PPO/RCWA-RL performance.

The table is descriptive protocol-design evidence, not a claim that a baseline satisfies a learned-method constraint on the final station test.

## 9. Water objective

For every requested eta, the optimization objective remains

\[
\min_\pi\;\mathbb{E}[I_{\mathrm{season,total}}]
\]

subject to the risk constraint above and all hard event/budget constraints.

Because every method receives the same fixed 45-mm preplant event,

\[
I_{\mathrm{season,total}}
=45\;\mathrm{mm}+I_{\mathrm{postplant,policy}},
\]

so minimizing total seasonal irrigation and minimizing controllable postplant irrigation are equivalent optimization problems. Published water totals must nevertheless include the fixed 45 mm so that results remain agronomically comparable with field irrigation totals.

## 10. Final-test lock

Before final evaluation, the following must already be frozen:

- Risk Contract v1;
- learned algorithm implementation;
- hyperparameters;
- training seeds;
- checkpoint/model-selection rule;
- selected learned checkpoints.

Only then may the final evaluator use station weather 2023-2025 and generate same-year Conventional-W100 station references. The final evaluation must not update parameters or select another checkpoint after those outcomes are observed.

## 11. Implementation files

- machine-readable contract: `configs/risk_contract_v1.json`;
- development Y_ref table: `configs/yield_reference_v1_development.json`;
- exact empirical estimator: `src/awm/risk/contract.py`;
- development risk audit: `src/awm/risk/development_audit.py`;
- unit tests: `tests/test_risk_contract.py`.
