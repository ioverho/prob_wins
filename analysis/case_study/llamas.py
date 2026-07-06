import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import pathlib
    from urllib.request import urlretrieve
    import json

    import marimo as mo
    import numpy as np

    import prob_wins

    return json, mo, np, pathlib, prob_wins, urlretrieve


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Download the evaluation files for model 1 and model 2
    """)
    return


@app.cell
def _(pathlib):
    model_1_data_url = "https://huggingface.co/datasets/evaleval/EEE_datastore/resolve/main/data/hellaswag/meta-llama/llama-3.2-90b-vision-instruct/15973299-9988-4eae-8545-f926c62e29cb_samples.jsonl"

    model_1_data_fp = pathlib.Path(
        "./data/hellaswag/llama-3.2-90b-vision-instruct/samples.jsonl"
    )

    model_1_data_fp.parent.mkdir(parents=True, exist_ok=True)
    return model_1_data_fp, model_1_data_url


@app.cell
def _(pathlib):
    model_2_data_url = "https://huggingface.co/datasets/evaleval/EEE_datastore/resolve/main/data/hellaswag/meta-llama/llama-3.3-70b-instruct/e5883c19-b73d-4c4a-a688-383973f05998_samples.jsonl"

    model_2_data_fp = pathlib.Path("./data/hellaswag/llama-3.3-70b-instruct/samples.jsonl")

    model_2_data_fp.parent.mkdir(parents=True, exist_ok=True)
    return model_2_data_fp, model_2_data_url


@app.cell
def _(
    model_1_data_fp,
    model_1_data_url,
    model_2_data_fp,
    model_2_data_url,
    urlretrieve,
):
    if not model_1_data_fp.exists():
        urlretrieve(
            url=model_1_data_url,
            filename=model_1_data_fp,
        )

    if not model_2_data_fp.exists():
        urlretrieve(
            url=model_2_data_url,
            filename=model_2_data_fp,
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read in the separate files
    """)
    return


@app.cell
def _(json, model_1_data_fp, model_2_data_fp):
    model_1_eval_results = dict()
    with model_1_data_fp.open("r") as f:
        for line in f:
            eval_row = json.loads(line)

            model_1_eval_results[eval_row["sample_id"]] = eval_row["evaluation"]["score"]

    model_2_eval_results = dict()
    with model_2_data_fp.open("r") as f:
        for line in f:
            eval_row = json.loads(line)

            model_2_eval_results[eval_row["sample_id"]] = eval_row["evaluation"]["score"]
    return model_1_eval_results, model_2_eval_results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Make sure the files are paired. Same sample result should be in the same index
    """)
    return


@app.cell
def _(model_1_eval_results, model_2_eval_results):
    intersection_samples = set(model_1_eval_results.keys()) & set(
        model_2_eval_results.keys()
    )
    return (intersection_samples,)


@app.cell
def _(intersection_samples, model_1_eval_results, model_2_eval_results):
    model_1_eval_results_paired = [
        model_1_eval_results[sample_id] for sample_id in intersection_samples
    ]

    model_2_eval_results_paired = [
        model_2_eval_results[sample_id] for sample_id in intersection_samples
    ]
    return model_1_eval_results_paired, model_2_eval_results_paired


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compute accuracy of the different models
    """)
    return


@app.cell
def _(model_1_eval_results_paired, model_2_eval_results_paired, np):
    model_1_acc = np.mean(model_1_eval_results_paired)
    model_2_acc = np.mean(model_2_eval_results_paired)

    print(f"Model 1 Accuracy: {model_1_acc:.2%}")
    print(f"Model 2 Accuracy: {model_2_acc:.2%}")
    return


@app.cell
def _(model_1_eval_results_paired, model_2_eval_results_paired, np):
    paired_confusion_matrix = np.zeros((2, 2))

    for id, count in zip(
        *np.unique(
            np.stack([model_1_eval_results_paired, model_2_eval_results_paired], axis=1),
            return_counts=True,
            axis=0,
        )
    ):
        paired_confusion_matrix[*id.astype(int)] = count

    paired_confusion_matrix.astype(int) / len(model_1_eval_results_paired)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Frequentist Comparison
    """)
    return


@app.cell
def _(model_1_eval_results_paired, model_2_eval_results_paired, prob_wins):
    pairwise_comparison_results = prob_wins.compare_paired_win_rates_frequentist(
        results=model_2_eval_results_paired,
        baseline_results=model_1_eval_results_paired,
        confidence_level=0.95,
    )
    return (pairwise_comparison_results,)


@app.cell
def _(pairwise_comparison_results):
    pairwise_comparison_results.prob_win, pairwise_comparison_results.prob_loss
    return


@app.cell
def _(pairwise_comparison_results):
    pairwise_comparison_results.test_statistic
    return


@app.cell
def _(pairwise_comparison_results):
    pairwise_comparison_results.test_p_val_one_sided
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bayesian
    """)
    return


@app.cell
def _(model_1_eval_results_paired, model_2_eval_results_paired, prob_wins):
    pairwise_comparison_results_bayesian = prob_wins.compare_paired_win_rates_bayesian(
        results=model_2_eval_results_paired,
        baseline_results=model_1_eval_results_paired,
        seed=0,
        prior_strategy="ones",
        num_samples=100000,
        min_sig_diff=2.00,
    )
    return (pairwise_comparison_results_bayesian,)


@app.cell
def _(pairwise_comparison_results_bayesian):
    pairwise_comparison_results_bayesian.p_sig_neg
    return


if __name__ == "__main__":
    app.run()
