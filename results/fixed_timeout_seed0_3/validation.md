# Fixed-Timeout Seeds 0-3 Validation

- PASS: exactly 120 unique method-seed-grid records.
- PASS: four observations per method and grid size.
- PASS: all source rows use seeds 0-3 and grids n=10-19.
- PASS: every timeout consumed the full 120-minute budget.
- PASS: method-specific success, status, and return-code combinations are valid.
- KNOWN ANOMALY: six ILP runs reached Gurobi's time limit without an incumbent and then returned code 1 while writing the missing objective (seed 0 n=19, seed 1 n=19, seed 2 n=18, seed 2 n=19, seed 3 n=18, seed 3 n=19).
- PASS: 20 solution-quality cells were generated with sample standard deviations using denominator r-1.
- PASS: timeout counts identify censored runtime values, including the mixed-censoring n=18 ILP median as a lower bound.
- NOTE: exploratory parallel seed-5 runs are excluded.
- NOTE: PPO seed 0 used an older runner revision than seeds 1-3; recorded hyperparameters match, but the historical runner diff is not version-controlled.
