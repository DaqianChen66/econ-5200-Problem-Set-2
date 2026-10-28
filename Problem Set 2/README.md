# econ-5200-Problem-Set-2

This project studies why a portfolio of A/B tests can appear successful while producing less business value than expected. I simulate a portfolio of 200 experiments with known true effects, apply the client's current `p < 0.05` shipping rule, measure the winner's curse, bootstrap the realised portfolio return, compare an alternative decision rule, and implement a sequential-testing extension.

## Project Structure

```text
Problem Set 2/
├── Econ_5200_PS2.ipynb
├── ai-appendix.md
├── README.md
└── src/
    └── experiment_portfolio.py
```

## Main Analysis

The simulated portfolio uses a mixture prior:

- 85% of experiments: `Normal(0.00, 0.01)`
- 15% of experiments: `Normal(0.08, 0.025)`
- 200 total experiments
- Sample size per arm: 500 to 20,000
- Outcome standard deviation: 1.0
- Random seed: 42

The client's current rule ships experiments with `p < 0.05`.

## Key Results

Under the current rule:

- 44 experiments ship
- 45.5% of shipped experiments come from the near-zero component
- 54.5% of shipped experiments come from the genuinely positive component
- Mean true effect among shipped experiments: 0.042
- Mean estimated effect among shipped experiments: 0.039
- Overall effect-magnitude inflation factor: 1.208
- Portfolio-level realised return: 1.854 effect units
- BCa 95% bootstrap interval: [1.158, 2.746]

The power-floor alternative requires `p < 0.05` and power of at least 80%. It ships 42 experiments, produces a realised return of 1.687, and has a false-ship rate of 47.6%. In this simulation, the alternative rule does not outperform the current rule.

## Sequential Testing

Phase 4 examines repeated monitoring using alpha-spending boundaries.

The first sequential procedure produced an empirical false positive rate of 3.9% across 1,000 pure-null streams. A revised version used fixed monitoring looks every 50 observations from 50 to 1,000 and produced a false positive rate of 2.0%, showing that the revised procedure is conservative relative to the nominal 5% level.

The full AI interaction, including both prompt versions and raw responses, is documented in `ai-appendix.md`.

## Python Module

`src/experiment_portfolio.py` contains:

- `simulate_portfolio()` — simulates an experiment portfolio with known truth
- `apply_rule()` — applies the threshold or power-floor decision rule
- `evaluate()` — reports realised return, ship count, and false-ship rate

The module includes docstrings, type hints, a command-line demonstration, and assertions.

## Main Takeaway

Statistical significance is not the same as business value. Selecting experiments because they cross a significance threshold can exaggerate observed effects and allow near-zero experiments to appear successful. Portfolio-level decisions should therefore be evaluated using realised true return and validated decision rules rather than the percentage of experiments declared statistically significant.
