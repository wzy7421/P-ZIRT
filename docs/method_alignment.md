# Method Alignment Notes

The manuscript-facing P-ZIRT claim should be framed as:

> a provenance-aware, zero-inflated, reliability-calibrated proxy-monitoring framework for partially decoded roadside V2X data.

Recommended boundaries:

- Treat TrafficFlow-derived queued-vehicle targets as proxy monitoring unless independent queue labels are available.
- Report road-lane group splits rather than relying only on random splits.
- Report probability calibration metrics such as Brier score, Brier skill, and ECE.
- Compare against no-skill, prevalence, lag, direct-regression, and hurdle baselines.
- Use road-lane cluster bootstrap or date-level block bootstrap when estimating uncertainty.

Avoid claiming:

- validated physical queue-length estimation without independent labels;
- universal superiority over all baselines when a baseline has a better metric;
- operational deployment readiness without data-permission, latency, and validation checks.
