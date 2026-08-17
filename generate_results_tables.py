"""
Regenerates results_tables.tex (and experiment_results_tables.pdf) from
baseline_results_10.csv: F1-score of Phase-1 reliable-negative selection and
Phase-2 classifier F1/accuracy, per dataset/model/percent_positive.

Usage:
    python generate_results_tables.py
    python generate_results_tables.py --csv baseline_results_dynrn_llgc.csv \\
        --tex results_tables_dynrn_llgc.tex --jobname experiment_results_tables_dynrn_llgc
"""
import argparse
import subprocess
import pandas as pd

MODEL_ORDER = ["RCSVM", "CCRNE", "PU_LP", "MCLS", "LP_PUL", "PULearningPC"]
MODEL_DISPLAY_NAME = {"PULearningPC": "PC-PUL"}
PCTS = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25]


def escape(name):
    return MODEL_DISPLAY_NAME.get(name, name).replace("_", r"\_")


def build_table(df, metric, caption, label):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{ll" + "c" * len(PCTS) + "}")
    lines.append(r"\hline")
    header = "Dataset & Model & " + " & ".join(f"{p:.2f}" for p in PCTS) + r" \\"
    lines.append(header)
    lines.append(r"\hline")

    for dataset in sorted(df["dataset"].unique()):
        ds_df = df[df["dataset"] == dataset]

        # mean per (model, pct) to find the per-column best (for bolding)
        stats = {}
        for model in MODEL_ORDER:
            for pct in PCTS:
                sub = ds_df[(ds_df["model"] == model) & (ds_df["percent_positive"] == pct)]
                if len(sub) == 0:
                    stats[(model, pct)] = (None, None)
                else:
                    stats[(model, pct)] = (sub[metric].mean(), sub[metric].std())

        best_per_pct = {}
        for pct in PCTS:
            vals = [(m, stats[(m, pct)][0]) for m in MODEL_ORDER if stats[(m, pct)][0] is not None]
            best_per_pct[pct] = max(vals, key=lambda x: x[1])[0] if vals else None

        for i, model in enumerate(MODEL_ORDER):
            row_ds = escape(dataset) if i == 0 else ""
            cells = []
            for pct in PCTS:
                mean, std = stats[(model, pct)]
                if mean is None:
                    cells.append("--")
                    continue
                std = 0.0 if pd.isna(std) else std
                text = f"{mean:.3f} $\\pm$ {std:.3f}"
                if best_per_pct[pct] == model:
                    text = r"\textbf{" + text + "}"
                cells.append(text)
            lines.append(f"{row_ds} & {escape(model)} & " + " & ".join(cells) + r" \\")
        lines.append(r"\hline")

    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Regenerate results tables PDF from a baseline_results CSV")
    parser.add_argument("--csv", type=str, default="baseline_results_10.csv", help="Input results CSV path")
    parser.add_argument("--tex", type=str, default="results_tables.tex", help="Output .tex path")
    parser.add_argument("--jobname", type=str, default="experiment_results_tables",
                        help="pdflatex jobname (output PDF is <jobname>.pdf)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "error" in df.columns:
        df = df[df["error"].isna()]

    table1 = build_table(
        df, "phase1_f1",
        r"F1-score da sele\c{c}\~ao de negativos confi\'aveis (Fase 1), "
        r"por dataset, modelo e fra\c{c}\~ao de positivos conhecidos.",
        "tab:phase1",
    )
    table2 = build_table(
        df, "phase2_f1",
        r"F1-score do classificador na Fase 2, por dataset, modelo e "
        r"fra\c{c}\~ao de positivos conhecidos.",
        "tab:phase2-f1",
    )
    table3 = build_table(
        df, "phase2_accuracy",
        r"Acur\'acia do classificador na Fase 2, por dataset, modelo e "
        r"fra\c{c}\~ao de positivos conhecidos.",
        "tab:phase2-acc",
    )

    doc = f"""\\documentclass[landscape]{{article}}
\\usepackage[margin=1cm]{{geometry}}
\\usepackage{{graphicx}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\pagestyle{{empty}}

\\begin{{document}}

{table1}

\\vspace{{1cm}}

{table2}

\\vspace{{1cm}}

{table3}

\\end{{document}}
"""
    with open(args.tex, "w") as f:
        f.write(doc)
    print(f"Wrote {args.tex}")

    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", f"-jobname={args.jobname}", args.tex],
        check=True, capture_output=True, text=True,
    )
    print(f"Compiled {args.jobname}.pdf")


if __name__ == "__main__":
    main()
