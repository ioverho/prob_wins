import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    import prob_wins

    return np, plt, prob_wins


@app.cell
def _(np):
    rng = np.random.default_rng(0)
    return (rng,)


@app.cell
def _(np, prob_wins, rng):
    num_outer_reps = 1000
    num_inner_reps = 100

    all_reps_prob_tie = []
    all_reps_mean_wr = []
    all_reps_mean_wr_lb = []
    all_reps_mean_wr_ub = []
    all_reps_mean_wo = []
    all_reps_mean_wo_lb = []
    all_reps_mean_wo_ub = []
    all_reps_mean_nb = []
    all_reps_mean_nb_lb = []
    all_reps_mean_nb_ub = []
    for alpha_tie_power in np.arange(0, 2):
        alpha_tie = 2**alpha_tie_power
        for rep_num in range(num_outer_reps):
            reps_prob_tie = []

            reps_mean_wr = []
            reps_mean_wr_lb = []
            reps_mean_wr_ub = []

            reps_mean_wo = []
            reps_mean_wo_lb = []
            reps_mean_wo_ub = []

            reps_mean_nb = []
            reps_mean_nb_lb = []
            reps_mean_nb_ub = []

            props = rng.dirichlet([1, 1, alpha_tie])

            for inner_rep_num in range(num_inner_reps):
                outcomes = rng.choice([[1, 0], [0, 1], [0, 0]], 100, p=props)

                outcomes_a = outcomes[:, 0]
                outcomes_b = outcomes[:, 1]

                results = prob_wins.compare_paired_win_rates_frequentist(
                    results=outcomes_a,
                    baseline_results=outcomes_b,
                    confidence_level=0.95,
                    score_interval_method="ac",
                )

                reps_prob_tie.append(results.prob_tie)

                reps_mean_wr.append(results.wr)
                reps_mean_wr_lb.append(results.wr_ci.lower)
                reps_mean_wr_ub.append(results.wr_ci.upper)

                reps_mean_wo.append(results.wo)
                reps_mean_wo_lb.append(results.wo_ci.lower)
                reps_mean_wo_ub.append(results.wo_ci.upper)

                reps_mean_nb.append(results.nb)
                reps_mean_nb_lb.append(results.nb_ci.lower)
                reps_mean_nb_ub.append(results.nb_ci.upper)

            all_reps_prob_tie.append(np.mean(reps_prob_tie))

            all_reps_mean_wr.append(np.mean(reps_mean_wr))
            all_reps_mean_wr_lb.append(np.mean(reps_mean_wr_lb))
            all_reps_mean_wr_ub.append(np.mean(reps_mean_wr_ub))

            all_reps_mean_wo.append(np.mean(reps_mean_wo))
            all_reps_mean_wo_lb.append(np.mean(reps_mean_wo_lb))
            all_reps_mean_wo_ub.append(np.mean(reps_mean_wo_ub))

            all_reps_mean_nb.append(np.mean(reps_mean_nb))
            all_reps_mean_nb_lb.append(np.mean(reps_mean_nb_lb))
            all_reps_mean_nb_ub.append(np.mean(reps_mean_nb_ub))

    return (
        all_reps_mean_nb,
        all_reps_mean_wo,
        all_reps_mean_wo_ub,
        all_reps_mean_wr,
        all_reps_prob_tie,
        results,
    )


@app.cell
def _(results):
    results
    return


@app.cell
def _(
    all_reps_mean_nb,
    all_reps_mean_wo,
    all_reps_mean_wr,
    all_reps_prob_tie,
    np,
    plt,
):
    fig, axes = plt.subplots(1, 3, squeeze=False, sharex=True)

    #
    axes[0, 0].scatter(np.array(all_reps_prob_tie), np.array(all_reps_mean_wr), alpha=0.75)

    axes[0, 0].set_yscale("log")

    axes[0, 0].set_ylim(1.5e-1, 1.5e1)

    axes[0, 0].set_title("WR")

    #
    axes[0, 1].scatter(np.array(all_reps_prob_tie), np.array(all_reps_mean_wo), alpha=0.75)

    axes[0, 1].set_ylim(1.5e-1, 1.5e1)

    axes[0, 1].set_yscale("log")

    axes[0, 1].set_title("WO")

    axes[0, 1].set_xlabel("p(A = B)")

    #
    axes[0, 2].scatter(np.array(all_reps_prob_tie), np.array(all_reps_mean_nb), alpha=0.75)

    axes[0, 2].set_ylim(-1.0, 1.0)

    axes[0, 2].set_title("NB")

    fig.tight_layout()

    fig
    return


@app.cell
def _(all_reps_mean_wo_ub, np):
    np.nanmin(all_reps_mean_wo_ub)
    return


if __name__ == "__main__":
    app.run()
