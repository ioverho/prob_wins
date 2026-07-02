import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

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
        ax.tick_params(axis="both", which="both", colors=color)

        ax.xaxis.label.set_color(color)
        ax.yaxis.label.set_color(color)

        ax.title.set_color(color)

        for _, spine in ax.spines.items():
            spine.set_color(color)


    return color_axis, np, plt, sns, update_theme


@app.cell
def _(update_theme):
    theme = "dark"

    update_theme(theme)
    return (theme,)


@app.cell
def _():
    ls_list = [
        "-",
        "--",
        ":",
        "-.",
        (5, (10, 3)),
    ]
    return (ls_list,)


@app.cell
def _(sns):
    cmap = sns.color_palette("viridis", as_cmap=True)
    return (cmap,)


@app.cell
def _(cmap, color_axis, color_text, ls_list, np, plt):
    fig, axes = plt.subplots(1, 3, figsize=(8.33, 3.13), squeeze=False)

    nb = np.linspace(-1, 1, 100)

    nb_to_wo = (1 + nb) / (1 - nb)

    axes[0, 0].plot(nb, nb_to_wo, color=color_text, ls="-")

    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlim(-1, 1)
    axes[0, 0].set_ylim(1e-2, 1e2)

    axes[0, 0].set_xlabel("NB")
    axes[0, 0].set_ylabel("WO")

    color_axis(axes[0, 0], color=color_text)

    axes[0, 0].spines["top"].set_visible(False)
    axes[0, 0].spines["right"].set_visible(False)


    def nb_to_wr_func(prob_tie: float):
        nb = np.linspace(-(1 - prob_tie - 1e-8), 1 - prob_tie - 1e-8, 1000)
        wr = (prob_tie - nb - 1) / (prob_tie + nb - 1)

        return nb, wr


    for p, ls in zip([0.1, 0.3, 0.5, 0.7, 0.9], ls_list):
        axes[0, 1].plot(*nb_to_wr_func(p), color=cmap(p), ls=ls)

    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlim(-1, 1)

    axes[0, 1].set_xlabel("NB")
    axes[0, 1].set_ylabel("WR")

    color_axis(axes[0, 1], color=color_text)

    axes[0, 1].spines["top"].set_visible(False)
    axes[0, 1].spines["right"].set_visible(False)


    def wo_to_wr_func(prob_tie: float):
        nb = np.linspace(-(1 - prob_tie - 1e-8), 1 - prob_tie - 1e-8, 1000)
        wr = (prob_tie - nb - 1) / (prob_tie + nb - 1)

        wo = (1 + nb) / (1 - nb)

        return wo, wr


    for p, ls in zip([0.1, 0.3, 0.5, 0.7, 0.9], ls_list):
        axes[0, 2].plot(
            *wo_to_wr_func(p),
            color=cmap(p),
            ls=ls,
            label=f"p(A=B)={p:.2f}",
        )

    axes[0, 2].set_xscale("log")
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_xlim((1 - 0.93) / (1 + 0.93), (1 + 0.93) / (1 - 0.93))

    axes[0, 2].set_xlabel("WO")
    axes[0, 2].set_ylabel("WR")

    color_axis(axes[0, 2], color=color_text)

    axes[0, 2].spines["top"].set_visible(False)
    axes[0, 2].spines["right"].set_visible(False)

    mappable = sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))

    colorbar = fig.colorbar(
        mappable=mappable,
        ax=axes[0, 2],
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

    fig.tight_layout()

    fig
    return (fig,)


@app.cell
def _(fig, theme):
    fig.savefig(
        f"./figures/transformations_{theme}.svg",
        transparent=True,
        pad_inches=0.0,
        bbox_inches="tight",
    )
    return


if __name__ == "__main__":
    app.run()
