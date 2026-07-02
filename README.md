<div style="text-align: center;" align="center">

<div style="text-align: center;" align="center">

<h1>Probabilistic Win Rate Comparisons</h1>

</div>

<div style="text-align: center;" align="center">

<a href="https://github.com/ioverho/prob_wins/actions/workflows/test.yaml" >
 <img src="https://github.com/ioverho/prob_wins/actions/workflows/test.yaml/badge.svg"/ alt="Tests status">
</a>

<a href="https://codecov.io/github/ioverho/prob_wins" >
 <img src="https://codecov.io/github/ioverho/prob_wins/graph/badge.svg?token=EU85JBF8M2"/ alt="Codecov report">
</a>

<a href="./LICENSE" >
 <img alt="GitHub License" src="https://img.shields.io/github/license/ioverho/prob_wins">
</a>

<a href="https://pypi.org/project/prob-conf-mat/" >
  <img alt="PyPI - Version" src="https://img.shields.io/pypi/v/prob_wins">
</a>

</div>

</div>

**`prob_wins`** is a Python package for performing pairwise comparisons of machine learning models with win rates. It computes the proportion of wins, losses and ties, generates several win rate effect size statistics, and can perform statistical inference tests.

## Installation

Installation can be done using from [pypi](https://pypi.org/project/prob-conf-mat/) can be done using `pip`:

```bash
pip install prob_wins
```

Or, if you're using [`uv`](https://docs.astral.sh/uv/), simply run:

```bash
uv add prob_wins
```

The project currently depends on the following packages:

<details>
  <summary>Dependency tree</summary>

```txt
prob-wins
├── jaxtyping
├── numpy
└── scipy
```

</details>

### Development Environment

This project was developed using [`uv`](https://docs.astral.sh/uv/). To install the development environment, simply clone this github repo:

```bash
git clone https://github.com/ioverho/prob_wins.git
```

And then run the `uv sync --dev` command:

```bash
uv sync --dev
```

The development dependencies should automatically install into the `.venv` folder.

## Documentation

Let's say you have some evaluation results from your model, `eval_results_model`, and from a baseline model, `eval_results_baseline`, and you want to assess how much better your model is than baseline. `prob_wins` provides two frameworks for doing this assessment.

A more detailed explanation and derivation of the various win rate statistics and testing is provided in an [accompanying blog post](https://www.ivoverhoeven.nl/blog/).

### Frequentist

Running the `prob_wins.compare_paired_win_rates_frequentist` method will return a `FrequentistComparisonResult` object that contains all necessary statistics for assessing your win rates *in a Frequentist framework*.

```python
import prob_wins

pairwise_comparison_results = prob_wins.compare_paired_win_rates_frequentist(
    results=eval_results_model,
    baseline_results=eval_results_baseline,
    confidence_level=0.95,
)
```

Specifically, it contains:
1. The proportion of times your model beat the baseline, and vice versa
   1. `prob_win` (`float`): Estimated probability that the system outperforms the baseline
   2. `prob_win_ci` (`ConfidenceInterval`): Confidence interval for `prob_win`
   3. `prob_loss` (`float`): Estimated probability that the system underperforms the baseline
   4. `prob_loss_ci` (`ConfidenceInterval`): Confidence interval for `prob_loss`
   5. `prob_tie` (`float`): Estimated probability of a tie
   6. `prob_tie_ci` (`ConfidenceInterval`): Confidence interval for `prob_tie`
2. Win rate statistics
   1. `wr` (`float`): Win ratio — ratio of wins to losses (prob_win / prob_loss)
   2. `wr_ci` (`ConfidenceInterval`): Confidence interval for win ratio
   3. `wo` (`float`): Win odds — (wins + 0.5*ties) / (losses + 0.5*ties)
   4.  `wo_ci` (`ConfidenceInterval`): Confidence interval for win odds
   5.  `nb` (`float`): Net benefit — difference of win and loss probabilities (prob_win - prob_loss)
   6.  `nb_ci` (`ConfidenceInterval`): Confidence interval for net benefit
3. Statistical test results
   1.  `test_statistic` (`float`): Sign test statistic: (wins - losses) / sqrt(wins + losses)
   2.  `test_p_val_one_sided` (`float`): One-sided p-value for the sign test (H1: system > baseline)
   3.  `test_p_val_two_sided` (`float`): Two-sided p-value for the sign test (H1: system =/= baseline)

### Bayesian

Running the `prob_wins.compare_paired_win_rates_bayesian` method will return a `BayesianComparisonResult` object that contains all necessary statistics for assessing your win rates *in a Bayesian framework*.

```python
import prob_wins

pairwise_comparison_results = prob_wins.compare_paired_win_rates_bayesian(
    results=eval_results_model,
    baseline_results=eval_results_baseline,
    seed=0,
    confidence_level=0.95,
    prior_strategy=[1, 1, 1],
    num_samples=10000,
    min_sig_diff=1.00,
)
```

The `BayesianComparisonResult` object contains largely the same elements (except that point statistics are now posterior medians, and confidence intervals become credibility intervals), except for the *p* values. These are replaced with:
1. `p_direction` (`float`): Probability of direction — posterior mass where test_statistic > 0
2. `p_sig_neg` (`float`): Posterior mass below the negative ROPE boundary (system is worse)
3. `p_sig_pos` (`float`): Posterior mass above the positive ROPE boundary (system is better)
4. `p_sig_bidirectional` (`float`): Total posterior mass outside the ROPE (p_sig_neg + p_sig_pos)

The ROPE is the region of practical equivalence, wherein the difference between the models might not be exactly 0, but it might as well be. The ROPE boundaries are determined by the `min_sig_diff` parameter.

## Citation

```bibtex
@software{ioverho_prob_wins,
    author = {Verhoeven, Ivo},
    license = {MIT},
    title = {{prob\_wins}},
    url = {https://github.com/ioverho/prob_wins}
}
```
