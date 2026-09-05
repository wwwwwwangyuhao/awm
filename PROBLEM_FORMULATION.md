# AWM Research Problem Formulation

## 1. Purpose

This repository studies **adaptive irrigation-water allocation for cotton under seasonal water scarcity and weather uncertainty**, using DSSAT as the crop–soil–water simulator and reinforcement learning (RL) as the decision engine.

The primary research problem is an **Agricultural Water Management problem**, not a generic RL benchmark:

> Given a limited seasonal irrigation allocation and uncertain in-season weather, how should irrigation water be distributed across the cotton growing season so that total irrigation is minimized while crop-yield reliability is maintained and all field-operational constraints are respected?

The first paper will focus on **irrigation only**. Nitrogen management is fixed to a validated recommended schedule so that improvements can be causally attributed to water-management decisions. Joint water–nitrogen management is reserved for a later extension.

---

## 2. Scientific Questions

The study is designed to answer four agricultural questions.

### Q1. Minimum water for reliable yield

What is the minimum seasonal irrigation required to maintain a specified fraction of reference cotton yield?

### Q2. Dynamic allocation under scarcity

When the seasonal water allocation is limited, how should scarce irrigation water be shifted among vegetative, flowering, boll formation/filling, and maturation stages?

### Q3. Robustness to climate variability

Can one management policy maintain the yield–water trade-off across dry, normal, wet, and hot–dry seasons without access to future weather information?

### Q4. Operational feasibility

Do the learned irrigation schedules remain beneficial after realistic irrigation-system constraints such as minimum event depth, maximum application capacity, minimum event interval, and seasonal water quota are enforced?

---

## 3. Decision Process

A growing season is represented by daily management decisions:

\[
t = 1,\ldots,T.
\]

DSSAT provides the crop–soil–weather transition dynamics. At decision day \(t\), the agent observes only information available up to that date, selects an irrigation action, and DSSAT generates the resulting crop and soil response.

Future DSSAT output must never be exposed to the policy. Seasonal summary variables such as final yield are available only after the episode terminates.

---

## 4. State

The policy state is

\[
s_t = [x_t,\; b_t,\; c_t,\; g_t,\; \tau_t,\; d_t,\; \eta],
\]

where:

- \(x_t\): current DSSAT crop–soil–weather observation;
- \(b_t\): remaining seasonal irrigation-budget fraction;
- \(c_t\): cumulative seasonal irrigation fraction;
- \(g_t\): crop phenological stage;
- \(\tau_t\): remaining fraction of the management horizon;
- \(d_t\): days since the last irrigation event;
- \(\eta\): requested yield-protection level.

The remaining-water state is defined as

\[
b_t = \frac{B-I_{1:t-1}}{B},
\]

where \(B\) is the seasonal irrigation allocation and

\[
I_{1:t-1}=\sum_{k=1}^{t-1} I_k.
\]

All state features must be computable from observations available at or before day \(t\). No forecast or future-weather feature is allowed in the canonical experiment unless a separate forecast-information experiment is explicitly defined.

---

## 5. Action

The irrigation action is modeled as a **hierarchical event–amount decision** rather than an unconstrained continuous value.

### 5.1 Event decision

\[
e_t \in \{0,1\},
\]

where \(e_t=0\) means no irrigation and \(e_t=1\) means an irrigation event occurs.

### 5.2 Irrigation amount

Conditional on \(e_t=1\), the policy chooses

\[
u_t \in [0,1],
\]

which is mapped to a feasible irrigation depth

\[
I_t = I_{\min} + u_t\left(I_{\max,t}-I_{\min}\right).
\]

If \(e_t=0\),

\[
I_t=0.
\]

This produces an exact no-irrigation action and prevents unrealistically small positive irrigation events.

---

## 6. Hard Agricultural Constraints

The canonical environment must enforce constraints directly rather than only penalizing their violation in the reward.

### 6.1 Seasonal irrigation allocation

\[
\sum_{t=1}^{T} I_t \le B.
\]

### 6.2 Event capacity

\[
I_t \le I_{\max,t}.
\]

\(I_{\max,t}\) must be derived from the actual irrigation-system capacity used in the study.

### 6.3 Minimum effective irrigation event

For a positive irrigation event,

\[
I_t \ge I_{\min}.
\]

\(I_{\min}\) must be based on actual field or irrigation-system practice, not selected for algorithmic convenience.

### 6.4 Minimum irrigation interval

If required by the real irrigation system,

\[
t_j-t_{j-1} \ge d_{\min}
\]

for consecutive irrigation events \(j-1\) and \(j\).

### 6.5 No future-water borrowing

At every decision,

\[
I_t \le B-I_{1:t-1}.
\]

Actions violating a hard constraint are masked or projected into the feasible action set before DSSAT execution. Constraint enforcement must be logged explicitly.

Numerical values for \(I_{\min}\), \(I_{\max}\), \(d_{\min}\), and \(B\) are **not frozen in this document** until justified by field-system specifications, local irrigation recommendations, or the experimental treatment design.

---

## 7. Reference Yield

The yield target is defined relative to a reference management condition.

Let

\[
Y_{\mathrm{ref}}(\omega)
\]

be the reference yield under weather realization \(\omega\). The canonical reference must be specified before formal training and then frozen. Candidate reference protocols include:

1. validated locally recommended irrigation management; or
2. a sufficiently watered agronomic reference constrained by the same irrigation-system capacity.

The reference definition must not use the learned policy and must not be changed after seeing final test results.

The target yield level is

\[
Y_{\mathrm{target}} = \eta Y_{\mathrm{ref}},
\]

with a canonical target such as \(\eta=0.95\) to be justified agronomically before the formal experiment.

---

## 8. Primary Optimization Problem

The principal management objective is to minimize seasonal irrigation while maintaining reliable yield:

\[
\min_{\pi}\; \mathbb{E}_{\omega,\pi}\left[I_{\mathrm{season}}\right]
\]

subject to

\[
\mathrm{LCVaR}_{\alpha}\left(Y_{\pi}\right)
\ge
\eta Y_{\mathrm{ref}}.
\]

Here \(\mathrm{LCVaR}_{\alpha}\) denotes the mean yield in the lower \(\alpha\)-tail of the yield distribution. It is used because low yield is the undesirable tail; using the lower-tail definition avoids ambiguity with the common upper-tail CVaR convention for losses.

For example, \(\alpha=0.10\) evaluates performance in the worst 10% of sampled seasonal outcomes.

The exact \(\alpha\) and \(\eta\) values must be pre-specified before final testing and subjected to sensitivity analysis.

---

## 9. RCWA-RL Algorithmic Formulation

The working algorithm name is **RCWA-RL: Risk-Constrained Water Allocation Reinforcement Learning**.

Its purpose is to solve the agricultural optimization problem above rather than maximize an arbitrary weighted reward.

### 9.1 Budget- and target-conditioned policy

One policy is trained across multiple water allocations and yield-protection targets:

\[
\pi_{\theta}(a_t\mid x_t,b_t,g_t,\tau_t,d_t,\eta).
\]

During training, seasonal water budgets \(B\) and yield targets \(\eta\) are sampled from pre-specified training ranges. Final evaluation must include both in-range and deliberately held-out conditions.

The intended property is that one frozen policy can adapt its irrigation behavior when the available seasonal water allocation or required yield-protection level changes.

### 9.2 Primal–dual yield-reliability constraint

Rather than choosing a fixed irrigation penalty coefficient, the constrained objective is represented with a non-negative dual variable \(\lambda\):

\[
\mathcal{L}(\pi,\lambda)
=
-\mathbb{E}[I_{\mathrm{season}}]
+
\lambda\left(
\mathrm{LCVaR}_{\alpha}(Y_{\pi})
-
\eta Y_{\mathrm{ref}}
\right).
\]

The dual variable is updated so that violation of the yield-reliability constraint increases the pressure to protect crop production, whereas persistent satisfaction of the constraint allows the policy to search for further irrigation savings.

The implementation must define the exact sign convention and optimizer updates in unit-tested code before formal training.

### 9.3 Distributional seasonal-yield critic

A critic estimates the distribution of terminal yield rather than only its expectation:

\[
Z_Y(s_t,b_t,\eta).
\]

This critic supports estimation of lower-tail yield risk and the yield-reliability constraint.

### 9.4 Budget-monotonic yield-value model

A separate or shared value head estimates attainable yield as a function of remaining water:

\[
V_Y(s_t,b_t).
\]

For identical crop–soil state \(s_t\), additional feasible remaining irrigation water should not reduce the attainable optimal yield. The model is regularized toward

\[
b_1>b_2
\Rightarrow
V_Y(s_t,b_1)\ge V_Y(s_t,b_2).
\]

One possible monotonicity penalty is

\[
\mathcal{L}_{\mathrm{mono}}
=
\max\left(
0,
V_Y(s_t,b-\Delta b)-V_Y(s_t,b)
\right).
\]

This is an algorithmic inductive bias and must be ablated experimentally; it is not assumed automatically correct under every learned approximation.

### 9.5 Dynamic marginal value of irrigation water

The yield-value model is used to estimate the state-dependent marginal value of remaining irrigation water:

\[
MVW_t
=
\frac{
V_Y(s_t,b_t)-V_Y(s_t,b_t-\Delta b)
}{\Delta b}.
\]

The intended physical interpretation is the expected marginal yield value of one additional unit of available irrigation water under the current crop–soil state and remaining season.

This quantity is primarily an **interpretability and water-allocation analysis variable**. It must not be presented as causal or economically exact unless separately validated.

---

## 10. Training Distribution

Training must expose the policy to variation in:

- historical training-weather years;
- seasonal irrigation allocation \(B\);
- yield-protection target \(\eta\);
- stochastic policy seeds.

Optional later robustness training may include uncertainty in DSSAT calibration parameters or representative soil profiles, but these extensions must not contaminate the canonical first experiment unless explicitly frozen in the experimental protocol.

The training, validation/model-selection, and final-test weather sets must be mutually exclusive.

---

## 11. Primary Agricultural Outcomes

Formal evaluation is based on physical agricultural outcomes, not RL return.

### 11.1 Yield

\[
Y\quad [\mathrm{kg\ ha^{-1}}].
\]

### 11.2 Seasonal irrigation

\[
I=\sum_t I_t\quad [\mathrm{mm}].
\]

### 11.3 Irrigation-water saving

Relative to baseline management:

\[
WS = 100\times\frac{I_{\mathrm{baseline}}-I_{\mathrm{policy}}}{I_{\mathrm{baseline}}}.
\]

### 11.4 Irrigation water productivity

Because 1 mm over 1 ha equals 10 m\(^3\),

\[
IWP=\frac{Y}{10I}\quad [\mathrm{kg\ m^{-3}}].
\]

### 11.5 Crop water productivity

When seasonal evapotranspiration is available and validated,

\[
CWP=\frac{Y}{10ET}\quad [\mathrm{kg\ m^{-3}}].
\]

### 11.6 Yield reliability

Report at minimum:

- mean yield;
- lower-tail CVaR yield;
- probability of meeting \(\eta Y_{\mathrm{ref}}\);
- yield coefficient of variation across independent weather years.

---

## 12. Mechanistic Water-Management Outcomes

The paper must explain *why* irrigation savings occur.

Required analyses include:

### 12.1 Growth-stage irrigation allocation

For each phenological stage \(k\),

\[
I_k = \sum_{t\in k} I_t.
\]

At minimum compare vegetative, reproductive/flowering, boll formation/filling, and maturation periods using DSSAT-consistent phenological definitions.

### 12.2 Root-zone water status

Analyze root-zone soil-water availability and crop water-stress indicators around irrigation events and critical growth stages.

### 12.3 Seasonal water balance

Where DSSAT outputs are available and validated, report

\[
P+I = ET + D + R + \Delta S,
\]

including precipitation \(P\), irrigation \(I\), evapotranspiration \(ET\), deep drainage \(D\), runoff \(R\), and soil-water-storage change \(\Delta S\).

The goal is to determine whether irrigation savings arise from reduced non-productive water loss, altered soil-water storage, changed crop water use, or a combination of mechanisms.

---

## 13. Mandatory Baselines

The final Agricultural Water Management experiment must not compare only RL algorithms.

At minimum include:

1. **Local conventional irrigation management**;
2. **ET-based irrigation scheduling** (for example an appropriately validated FAO-56/ETc protocol);
3. **soil-water-threshold feedback irrigation**;
4. **a standard RL irrigation baseline** using the same DSSAT state information and operational constraints;
5. **RCWA-RL**.

Algorithmic baselines may be added, but they do not replace agricultural-management baselines.

All methods must use the same weather, soil, cultivar, planting date, seasonal water availability, irrigation-system capacity, decision horizon, and information set unless the experiment explicitly studies one of those factors.

---

## 14. Canonical Water-Scarcity Evaluation

The formal study will evaluate multiple seasonal irrigation allocations, provisionally:

\[
W_{100},\quad W_{80},\quad W_{60},
\]

where each level is defined relative to a frozen agronomic/reference water allocation rather than an arbitrary neural-network scale.

The exact treatment depths must be derived from the study region and documented before final runs.

The principal response is the **yield–irrigation frontier**:

\[
I_{\mathrm{season}} \leftrightarrow Y.
\]

A superior management policy should either:

- obtain greater yield for the same available irrigation water; or
- require less irrigation to meet the same yield-reliability target.

---

## 15. Climate-Robustness Evaluation

Final weather years must be classified using pre-defined climatic criteria into relevant groups such as:

- dry;
- normal;
- wet;
- hot–dry compound conditions.

Climate classification must use meteorological variables independent of algorithm performance.

For every weather year, all management methods are evaluated under exactly the same weather realization so that paired differences can be calculated:

\[
\Delta Y_y,\quad \Delta I_y,\quad \Delta IWP_y.
\]

The final paper must report performance distributions and effect sizes across independent weather years rather than treating daily time steps as statistical replicates.

---

## 16. Statistical Unit and Replication

A daily RL decision is not an independent experimental replicate.

For seasonal outcomes, valid units include independent weather years, site-years, soils where explicitly studied, independent RL training seeds, and field blocks in real-world validation.

All stochastic RL methods must use multiple independent training seeds. The canonical number of seeds will be fixed in the experiment protocol based on compute budget and variance observed in pilot runs; no best-seed-only reporting is permitted.

Final comparisons should be paired by weather year/site and report effect sizes and confidence intervals in addition to significance tests where appropriate.

---

## 17. Algorithmic Ablations

The following ablations are required to distinguish agricultural value from algorithmic complexity:

1. **No yield-risk constraint:** expected-yield objective only;
2. **No target conditioning:** separate fixed-target policy;
3. **No budget conditioning:** policy without explicit remaining-water state;
4. **No distributional critic:** expectation-only yield critic;
5. **No monotonic regularization:** remove \(\mathcal{L}_{\mathrm{mono}}\);
6. **No hierarchical event–amount action:** standard continuous irrigation action;
7. **No operational projection:** idealized simulator-only action space, reported only as a diagnostic and never as the practical result.

Ablations must be evaluated using physical outcomes \(Y\), \(I\), IWP, and yield reliability, not only training return.

---

## 18. Claims the First Paper May Make

If supported by the experiments, the first paper may claim that the proposed framework:

- identifies lower-irrigation strategies that meet a pre-specified cotton yield-reliability target;
- dynamically reallocates limited seasonal irrigation water across growth stages;
- adapts one policy to multiple water allocations and yield-protection targets;
- improves the yield–irrigation or yield–reliability frontier relative to conventional, ET-based, soil-threshold, and standard-RL management;
- remains effective under independent climate conditions and realistic irrigation-system constraints;
- provides interpretable estimates of state-dependent marginal irrigation-water value.

---

## 19. Claims the First Paper Must Not Make Without Additional Evidence

The first paper must not claim:

- universal superiority across crops, regions, or irrigation systems;
- direct field deployability without independent field validation;
- causal economic value of the learned marginal-water-value estimate without economic validation;
- real-world water savings solely from simulator results;
- water–nitrogen co-optimization, because nitrogen is fixed in the first canonical study;
- robustness to future climate change unless explicit future-climate scenarios are evaluated;
- that DSSAT uncertainty has been solved unless a formal parameter/model uncertainty experiment is performed.

---

## 20. Frozen Scope for Version 1

The first canonical version of the `awm` project is therefore:

> **Cotton irrigation-only management under finite seasonal water allocation, using a DSSAT-based daily closed-loop environment and a risk-constrained, budget- and target-conditioned RL policy. Nitrogen is fixed. Future weather is hidden. Final success is judged by yield reliability, seasonal irrigation use, irrigation water productivity, growth-stage water allocation, and robustness across independent weather conditions under realistic irrigation constraints.**

Any later change to this scope must be documented explicitly rather than silently changing the optimization problem after experimental results are observed.

---

## 21. Parameters Still Requiring Empirical Freezing

The following quantities are intentionally left unresolved until supported by field-system specifications, agronomic recommendations, calibration data, or pilot-analysis evidence:

- exact seasonal reference irrigation \(B_{\mathrm{ref}}\);
- water-scarcity treatment depths for W100/W80/W60;
- minimum event depth \(I_{\min}\);
- maximum event depth/system capacity \(I_{\max}\);
- minimum irrigation interval \(d_{\min}\);
- exact reference-yield protocol \(Y_{\mathrm{ref}}\);
- canonical yield-protection target \(\eta\);
- lower-tail risk level \(\alpha\);
- number of independent RL training seeds;
- exact climatic classification thresholds;
- final agricultural baseline schedules;
- whether DSSAT parameter/soil uncertainty is included in the canonical main study or a robustness extension.

These values must be justified and frozen in an experimental protocol before the formal test set is evaluated.