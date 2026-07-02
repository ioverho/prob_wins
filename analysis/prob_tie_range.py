import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    import prob_wins

    themes = {
        "light": {
            "color_text": "#181a20",
            "color_background": "#fafafa",
            "color_primary": "#1a8fe3",
            "color_2": "#ff73b4",
            "color_3": "#4fcb8c",
            "color_4": "#fe9d3a",
        },
        "dark": {
            "color_text": "#fafafa",
            "color_background": "#181a20",
            "color_primary": "#1a8fe3",
            "color_2": "#ff73b4",
            "color_3": "#4fcb8c",
            "color_4": "#fe9d3a",
        },
    }


    def update_theme(theme: str) -> None:
        globals()["theme"] = theme
        globals()["color_text"] = themes[theme]["color_text"]
        globals()["color_background"] = themes[theme]["color_background"]
        globals()["color_primary"] = themes[theme]["color_primary"]
        globals()["color_2"] = themes[theme]["color_2"]
        globals()["color_3"] = themes[theme]["color_3"]
        globals()["color_4"] = themes[theme]["color_4"]


    def color_axis(ax, color: str):
        ax.tick_params(axis="both", colors=color)

        ax.xaxis.label.set_color(color)
        ax.yaxis.label.set_color(color)

        ax.title.set_color(color)

        for _, spine in ax.spines.items():
            spine.set_color(color)


    return color_axis, np, plt, prob_wins, update_theme


@app.cell
def _(update_theme):
    theme = "light"

    update_theme(theme)
    return (theme,)


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

    all_reps_prob_tie = np.array(all_reps_prob_tie)
    all_reps_mean_wr = np.array(all_reps_mean_wr)
    all_reps_mean_wr_lb = np.array(all_reps_mean_wr_lb)
    all_reps_mean_wr_ub = np.array(all_reps_mean_wr_ub)
    all_reps_mean_wo = np.array(all_reps_mean_wo)
    all_reps_mean_wo_lb = np.array(all_reps_mean_wo_lb)
    all_reps_mean_wo_ub = np.array(all_reps_mean_wo_ub)
    all_reps_mean_nb = np.array(all_reps_mean_nb)
    all_reps_mean_nb_lb = np.array(all_reps_mean_nb_lb)
    all_reps_mean_nb_ub = np.array(all_reps_mean_nb_ub)
    return (
        all_reps_mean_nb,
        all_reps_mean_wo,
        all_reps_mean_wr,
        all_reps_prob_tie,
    )


@app.cell
def _(
    all_reps_mean_nb,
    all_reps_mean_wo,
    all_reps_mean_wr,
    all_reps_prob_tie,
    color_axis,
    color_text,
    plt,
):
    fig, axes = plt.subplots(1, 3, figsize=(8.33, 3.13), squeeze=False, sharex=True)

    #
    axes[0, 0].scatter(
        all_reps_prob_tie,
        all_reps_mean_wr,
        alpha=1.00,
        c=all_reps_prob_tie,
        vmin=0.0,
        vmax=1.0,
    )

    axes[0, 0].set_yscale("log")

    axes[0, 0].set_ylim(1.0e-1, 1.0e1)
    axes[0, 0].set_xlim(0, 1)

    axes[0, 0].set_title("WR")

    axes[0, 0].set_xlabel("p(A = B)")

    color_axis(axes[0, 0], color=color_text)

    axes[0, 0].spines["top"].set_visible(False)
    axes[0, 0].spines["right"].set_visible(False)

    axes[0, 0].tick_params(axis="both", which="both", color=color_text)

    #
    axes[0, 1].scatter(
        all_reps_prob_tie,
        all_reps_mean_wo,
        alpha=1.00,
        c=all_reps_prob_tie,
        vmin=0.0,
        vmax=1.0,
    )

    axes[0, 1].set_ylim(1.0e-1, 1.0e1)
    axes[0, 1].set_xlim(0, 1)

    axes[0, 1].set_yscale("log")

    axes[0, 1].set_title("WO")

    axes[0, 1].set_xlabel("p(A = B)")

    color_axis(axes[0, 1], color=color_text)

    axes[0, 1].spines["top"].set_visible(False)
    axes[0, 1].spines["right"].set_visible(False)

    axes[0, 1].tick_params(
        axis="both",
        which="both",
        color=color_text,
    )

    #
    scatter_plot = axes[0, 2].scatter(
        all_reps_prob_tie,
        all_reps_mean_nb,
        alpha=1.00,
        c=all_reps_prob_tie,
        vmin=0.0,
        vmax=1.0,
    )

    axes[0, 2].set_ylim(-1.0, 1.0)
    axes[0, 2].set_xlim(0, 1)

    axes[0, 2].set_title("NB")

    axes[0, 2].set_xlabel("p(A = B)")

    color_axis(axes[0, 2], color=color_text)

    axes[0, 2].spines["top"].set_visible(False)
    axes[0, 2].spines["right"].set_visible(False)

    axes[0, 2].tick_params(
        axis="both",
        which="both",
        color=color_text,
    )

    colorbar_1 = fig.colorbar(
        scatter_plot,
    )

    colorbar_1.set_label(
        r"$p(A=B)$",
        fontsize=11,
        color=color_text,
    )

    colorbar_1.set_ticks(
        ticks=[0.1, 0.3, 0.5, 0.7, 0.9],
        labels=[0.1, 0.3, 0.5, 0.7, 0.9],
        fontsize=11,
        color=color_text,
    )

    colorbar_1.outline.set_color(color_text)

    colorbar_1.ax.yaxis.set_tick_params(color=color_text)

    fig.tight_layout()

    fig
    return (fig,)


@app.cell
def _(fig, theme):
    fig.savefig(
        f"./figures/statistics_by_prob_tie_{theme}.svg",
        transparent=True,
        pad_inches=0.0,
        bbox_inches="tight",
    )
    return


@app.cell
def _(
    all_reps_mean_wo,
    all_reps_mean_wr,
    all_reps_prob_tie,
    color_axis,
    color_text,
    np,
    plt,
):
    fig_wr_wo_correspondence, axes_wr_wo_correspondence = plt.subplots(
        1,
        1,
        figsize=(8.33, 3.13),
        squeeze=False,
    )

    sort_idx = np.argsort(all_reps_prob_tie)

    axes_wr_wo_correspondence_scatter = axes_wr_wo_correspondence[0, 0].scatter(
        all_reps_mean_wo[sort_idx],
        all_reps_mean_wr[sort_idx],
        zorder=1,
        c=all_reps_prob_tie[sort_idx],
        alpha=1.0,
        vmin=0,
        vmax=1,
    )

    axes_wr_wo_correspondence[0, 0].set_xscale("log")
    axes_wr_wo_correspondence[0, 0].set_yscale("log")

    axes_wr_wo_correspondence[0, 0].set_xlim(1e-2, 1e2)
    axes_wr_wo_correspondence[0, 0].set_ylim(1e-2, 1e2)

    axes_wr_wo_correspondence[0, 0].plot(
        [1e-2, 1e2],
        [1e-2, 1e2],
        zorder=2,
        c=color_text,
        ls="--",
        linewidth=1,
    )

    axes_wr_wo_correspondence[0, 0].set_ylabel("WR")
    axes_wr_wo_correspondence[0, 0].set_xlabel("WO")

    color_axis(axes_wr_wo_correspondence[0, 0], color=color_text)

    axes_wr_wo_correspondence[0, 0].spines["top"].set_visible(False)
    axes_wr_wo_correspondence[0, 0].spines["right"].set_visible(False)

    axes_wr_wo_correspondence[0, 0].tick_params(
        axis="both",
        which="both",
        color=color_text,
    )

    colorbar = fig_wr_wo_correspondence.colorbar(
        axes_wr_wo_correspondence_scatter,
    )

    colorbar.set_label(
        r"$p(A=B)$",
        fontsize=11,
        color=color_text,
    )

    colorbar.set_ticks(
        ticks=[0.1, 0.3, 0.5, 0.7, 0.9],
        labels=[0.1, 0.3, 0.5, 0.7, 0.9],
        fontsize=11,
        color=color_text,
    )

    colorbar.outline.set_color(color_text)

    colorbar.ax.yaxis.set_tick_params(color=color_text)

    fig_wr_wo_correspondence.tight_layout()

    fig_wr_wo_correspondence
    return (fig_wr_wo_correspondence,)


@app.cell
def _(fig_wr_wo_correspondence, theme):
    fig_wr_wo_correspondence.savefig(
        f"./figures/wo_wr_correspondence_by_prob_tie_{theme}.svg",
        transparent=True,
        pad_inches=0.0,
        bbox_inches="tight",
    )
    return


if __name__ == "__main__":
    app.run()
