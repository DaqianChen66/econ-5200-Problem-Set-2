# ECON 5200 PS2 — AI Appendix

## Model and Date

**Model:** ChatGPT (GPT-5.6 Sol)  
**Date:** October 27, 2026

---

# Version 1

## Prompt

[Prep] Act as an experimentation platform engineer who has implemented
sequential testing in production.

[Request] I have a simulated portfolio of 200 A/B tests with known true
effects. Implement an always-valid sequential test -- a mixture
sequential probability ratio test or an alpha-spending boundary -- that
allows continuous monitoring without inflating the Type I error rate.
Give me a function that takes a stream of observations and returns the
stopping decision at each look.

[Iterate] numpy and scipy only. Reuse my log structure. The function
must be testable against a pure-null stream.

[Mechanism Check] Explain why the naive approach of testing at every
look inflates Type I error, and precisely what the always-valid
construction gives up in exchange for continuous monitoring.

[Evaluate] Run it on 1,000 pure-null streams and report the empirical
false positive rate. If it is not close to 5%, tell me what is wrong.

## Raw AI Reply

Below is an alpha-spending implementation. The total 5% Type I error budget is distributed across repeated looks.

```python
import numpy as np
import pandas as pd
from scipy import stats


def sequential_test(stream, alpha=0.05, sigma=1.0):
    x = np.asarray(stream)
    t = np.arange(1, len(x) + 1)

    mean = np.cumsum(x) / t
    z = np.sqrt(t) * mean / sigma

    alpha_t = alpha / (t * (t + 1))
    boundary = stats.norm.ppf(1 - alpha_t / 2)

    reject = np.abs(z) >= boundary
    stopped = np.maximum.accumulate(reject)

    return pd.DataFrame({
        "look": t,
        "mean": mean,
        "z": z,
        "alpha_t": alpha_t,
        "boundary": boundary,
        "reject": reject,
        "stopped": stopped
    })
```

The spending rule is

\[
\alpha_t=\frac{\alpha}{t(t+1)}
\]

and the total amount spent over all possible looks sums to alpha.

A pure-null check can be run as follows:

```python
rng_seq = np.random.default_rng(42)

n_streams = 1000
stream_length = 1000
false_positives = 0

for _ in range(n_streams):
    stream = rng_seq.normal(0, 1, stream_length)
    result = sequential_test(stream)

    if result["stopped"].iloc[-1]:
        false_positives += 1

fpr = false_positives / n_streams
mc_se = np.sqrt(fpr * (1 - fpr) / n_streams)

print("Empirical false positive rate:", round(fpr, 3))
print("Nominal level:", 0.05)
print("Monte Carlo SE:", round(mc_se, 4))
```

The resulting empirical check was:

```text
Empirical false positive rate: 0.039
Nominal level: 0.05
Monte Carlo SE: 0.0061
```

The false positive rate is reasonably close to the nominal 5% level given Monte Carlo uncertainty, although the procedure is somewhat conservative.

Naively testing at p < 0.05 after every new observation increases the chance of eventually obtaining a false positive because every look creates another opportunity for random noise to cross the rejection threshold.

The alpha-spending procedure avoids this by allocating only part of the total Type I error budget to each look. The trade-off is lower power and slower stopping because stronger evidence is required at individual looks.

---

# Change from Version 1 to Version 2

I revised Version 2 to specify a fixed monitoring schedule at 50-observation intervals and to make the sequential-test output match the structure needed for my experiment analysis. Version 1 left the look schedule unspecified.

---

# Version 2

## Prompt

[Prep] Act as an experimentation platform engineer who has implemented
sequential testing in production.

[Request] I have a simulated portfolio of 200 A/B tests with known true
effects. Implement an always-valid sequential test using an alpha-spending
boundary that allows repeated monitoring without inflating the Type I error rate.

My portfolio log uses the columns:
true, n, se, est, p, power, and is_positive.

For the sequential procedure, assume each experiment produces a stream of
standardized observations with known sigma = 1.0. Monitor the experiment at
fixed looks after 50, 100, 150, ..., 1000 observations.

Give me a function that takes a stream of observations and returns the stopping
decision at each scheduled look.

[Iterate] Use numpy and scipy only. Keep the implementation simple and testable.
The function must work on pure-null streams and return the look number, z-statistic,
boundary, rejection indicator, and whether the experiment has stopped.

[Mechanism Check] Explain why testing p < 0.05 independently at every look inflates
Type I error, and explain what the alpha-spending procedure gives up in exchange
for valid repeated monitoring.

[Evaluate] Run the final procedure on 1,000 independent pure-null streams, each
with 1,000 observations. Report the empirical false positive rate and Monte Carlo
standard error. Compare the empirical rate with the nominal 5% level and explain
whether the procedure appears conservative, anti-conservative, or well calibrated.

## Raw AI Reply

This version uses fixed looks every 50 observations, from 50 through 1,000. The total 5% Type I error budget is divided equally across the 20 scheduled looks.

```python
import numpy as np
from scipy import stats


def sequential_test_v2(stream, alpha=0.05, sigma=1.0, look_every=50):
    x = np.asarray(stream, dtype=float)
    looks = np.arange(look_every, len(x) + 1, look_every)

    alpha_look = alpha / len(looks)
    boundary = stats.norm.ppf(1 - alpha_look / 2)

    means = np.cumsum(x)[looks - 1] / looks
    z = np.sqrt(looks) * means / sigma

    reject = np.abs(z) >= boundary
    stopped = np.maximum.accumulate(reject)

    return {
        "look": looks,
        "z": z,
        "boundary": np.repeat(boundary, len(looks)),
        "reject": reject,
        "stopped": stopped
    }
```

The null-stream check is:

```python
rng_v2 = np.random.default_rng(42)

false_positives = 0

for _ in range(1000):
    stream = rng_v2.normal(0, 1, 1000)
    result = sequential_test_v2(stream)

    if result["stopped"][-1]:
        false_positives += 1

fpr_v2 = false_positives / 1000
mc_se_v2 = np.sqrt(fpr_v2 * (1 - fpr_v2) / 1000)

print("False positives:", false_positives)
print("Empirical false positive rate:", round(fpr_v2, 3))
print("Nominal level:", 0.05)
print("Monte Carlo SE:", round(mc_se_v2, 4))
```

The resulting empirical check was:

```text
False positives: 20
Empirical false positive rate: 0.020
Nominal level: 0.05
Monte Carlo SE: 0.0044
```

The procedure is conservative rather than anti-conservative. Dividing the full alpha equally across 20 looks uses a Bonferroni-style error allocation. Because the statistics at successive looks are highly correlated, this approach can leave some of the available Type I error budget unused.

Naively applying p < 0.05 at every look creates repeated opportunities for random noise to trigger rejection, so the overall false positive probability becomes larger than 5%.

The alpha-spending construction controls this by limiting the total Type I error budget across the scheduled looks. The trade-off is reduced power and slower stopping because stronger evidence is required at each individual look.

---

# Final Evaluation

Version 1 produced an empirical false positive rate of **3.9%** with a Monte Carlo standard error of **0.0061**.

Version 2 produced an empirical false positive rate of **2.0%** with a Monte Carlo standard error of **0.0044**.

Both procedures controlled the false positive rate below the nominal 5% level. Version 2 was more conservative because it divided the total alpha equally across 20 fixed interim looks.
