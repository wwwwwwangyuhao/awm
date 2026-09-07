# DSSAT Summary IRCM reporting erratum v1

Formal learned-method runs originally reconciled the exact irrigation ledger directly against `Summary.OUT::IRCM` using a 0.1-mm tolerance. Real DSSAT probes show that the custom DSSAT 4.8.5 build reports `IRCM` at integer-mm resolution with half-up rounding: 74.5 mm is reported as 75 mm, 89.2 mm as 89 mm, and 539.7 mm as 540 mm.

This is a reporting-domain issue, not a management-write or water-budget error. The corrected audit keeps the exact controller/COX ledger unchanged, converts the exact expected seasonal irrigation to the same 1-mm Summary reporting domain, and applies the existing tolerance only after that transformation. The tolerance is not widened.

Examples locked by regression tests:

- exact 539.7 mm -> expected Summary 540 mm; observed 540 mm: PASS;
- exact 539.4 mm -> expected Summary 539 mm; observed 539 mm: PASS;
- exact 539.4 mm -> expected Summary 539 mm; observed 540 mm: FAIL;
- exact 74.5 mm -> expected Summary 75 mm; observed 75 mm: PASS.

Formal PPO and PPO-Lagrangian runs started before this erratum are quarantined rather than continued under mixed commits. They must restart from update 0 under a commit containing this fix. No 2023-2025 station final-test data were used to discover or define the erratum.
