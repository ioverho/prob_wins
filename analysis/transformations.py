import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    return np, plt


@app.cell
def _(np, plt):
    fig, axes = plt.subplots(1, 3, squeeze=False)

    nb = np.linspace(-1, 1, 100)

    nb_to_wo = (1 + nb) / (1 - nb)

    axes[0, 0].plot(nb, nb_to_wo)

    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlim(-1, 1)

    axes[0, 0].set_xlabel("NB")
    axes[0, 0].set_ylabel("WO")


    def nb_to_wr_func(prob_tie: float):
        nb = np.linspace(-(1 - prob_tie - 1e-8), 1 - prob_tie - 1e-8, 1000)
        wr = (prob_tie - nb - 1) / (prob_tie + nb - 1)

        return nb, wr


    for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
        axes[0, 1].plot(*nb_to_wr_func(p))

    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlim(-1, 1)

    axes[0, 1].set_xlabel("NB")
    axes[0, 1].set_ylabel("WR")


    def wo_to_wr_func(prob_tie: float):
        nb = np.linspace(-(1 - prob_tie - 1e-8), 1 - prob_tie - 1e-8, 1000)
        wr = (prob_tie - nb - 1) / (prob_tie + nb - 1)

        wo = (1 + nb) / (1 - nb)

        return wo, wr


    for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
        axes[0, 2].plot(*wo_to_wr_func(p))

    axes[0, 2].set_xscale("log")
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_xlim((1 - 0.93) / (1 + 0.93), (1 + 0.93) / (1 - 0.93))

    axes[0, 2].set_xlabel("WO")
    axes[0, 2].set_ylabel("WR")

    fig.tight_layout()

    fig
    return


@app.cell
def _():
    (1 + 0.9) / (1 - 0.9), 
    return


if __name__ == "__main__":
    app.run()
