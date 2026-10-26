"""Aggregate raw results into summary CSVs and figures.

Reads results/raw/, writes results/summary/*.csv and results/figures/*.png, and
the gitignored disagreement review file for the manual error categorisation.
Each suite is summarised only if its raw files exist, so this can run after any
subset of suites. Behavioural summaries read the committed (redacted) files, so
the CSVs rebuild from a clean clone; the review file needs the local unredacted
files and is skipped without them.
"""

import csv
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from faithful_compression.stats.paired import merge_by_item, read_jsonl  # noqa: E402
from faithful_compression.stats.tables import (  # noqa: E402
    agreement_summary,
    behavioural_summary,
    capability_summary,
    preregistered_claim,
    reliability_summary,
    systems_summary,
)

RAW_DIR = Path("results/raw")
SUMMARY_DIR = Path("results/summary")
FIGURES_DIR = Path("results/figures")

# Reference palette (dataviz skill), validated all-pairs for 3 slots, light mode.
PRECISION_COLOURS = {"f16": "#2a78d6", "q8": "#eb6834", "q4": "#1baf7a"}
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SAME_CELL, CHANGED_CELL = "#f0efec", "#6da7ec"
TASK_TITLES = {"gsm8k": "GSM8K (maths, n=200)", "arc_challenge": "ARC-Challenge (science, n=300)"}
PAIRS = [("f16", "q8"), ("f16", "q4"), ("q8", "q4")]


def load_rows(pattern: str, raw_dir: Path) -> list[dict]:
    """Concatenate every raw file matching pattern, excluding unredacted copies."""
    files = sorted(p for p in raw_dir.glob(pattern) if "_unredacted" not in p.name)
    return [row for path in files for row in read_jsonl(path)]


def style_axes(ax: plt.Axes) -> None:
    """Recessive hairline chrome: no top/right spines, muted ticks, light grid."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def figure_pareto(systems_rows: list[dict], capability, out: Path) -> None:
    """Figure 1: decode speed vs accuracy per task; marker area follows GPU memory."""
    speed, vram = {}, {}
    for precision in PRECISION_COLOURS:
        rows = [r for r in systems_rows if r["precision"] == precision]
        speed[precision] = sum(r["decode_tok_s"] for r in rows) / len(rows)
        vram[precision] = rows[0]["peak_vram_mb"] / 1024
    tasks = [t for t in TASK_TITLES if t in set(capability["task"])]
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.2 * len(tasks), 4.4), facecolor=SURFACE)
    for ax, task in zip(axes if len(tasks) > 1 else [axes], tasks, strict=True):
        style_axes(ax)
        for precision, colour in PRECISION_COLOURS.items():
            row = capability[(capability["task"] == task) & (capability["precision"] == precision)]
            mean = float(row["mean_score"].iloc[0])
            low, high = float(row["ci_lower"].iloc[0]), float(row["ci_upper"].iloc[0])
            ax.errorbar(
                speed[precision],
                mean,
                yerr=[[mean - low], [high - mean]],
                fmt="none",
                ecolor=colour,
                elinewidth=2,
                capsize=0,
                zorder=2,
            )
            ax.scatter(
                speed[precision],
                mean,
                s=40 * vram[precision],
                color=colour,
                edgecolors=SURFACE,
                linewidths=2,
                zorder=3,
                label=precision,
            )
            ax.annotate(
                f"{precision}  {mean:.1%}\n{vram[precision]:.1f} GB",
                (speed[precision], high),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color=INK,
            )
        ax.set_title(TASK_TITLES[task], loc="left", fontsize=11, color=INK)
        ax.set_xlabel("decode speed (tokens/s)", color=INK_2, fontsize=9)
        ax.set_ylabel("accuracy, 95% CI", color=INK_2, fontsize=9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_xlim(50, 240)
        ax.set_ylim(0.55, 0.92)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=colour, label=precision)
        for precision, colour in PRECISION_COLOURS.items()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        labelcolor=INK_2,
        bbox_to_anchor=(0.5, -0.02),
    )
    shift = 100 * float((capability["delta_vs_f16"].abs()).max())
    fig.suptitle(
        f"Q4 decodes {speed['q4'] / speed['f16']:.1f}x faster in "
        f"{vram['q4'] / vram['f16']:.0%} of the GPU memory; "
        f"mean accuracy moves {shift:.1f} points or less",
        x=0.01,
        y=0.985,
        ha="left",
        fontsize=12,
        color=INK,
    )
    fig.text(
        0.01,
        0.935,
        "Marker area proportional to GPU memory. Llama-3.2-3B-Instruct, RX 9070 XT.",
        va="top",
        fontsize=9,
        color=INK_2,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def figure_disagreement(capability_rows: list[dict], out: Path) -> None:
    """Figure 2: which items change their extracted answer, per precision pair."""
    tasks = [t for t in TASK_TITLES if any(r["task"] == t for r in capability_rows)]
    fig, axes = plt.subplots(
        len(tasks), 1, figsize=(11, 1.9 * len(tasks) + 0.9), facecolor=SURFACE
    )
    for ax, task in zip(axes if len(tasks) > 1 else [axes], tasks, strict=True):
        by_p: dict[str, list[dict]] = {}
        for row in capability_rows:
            if row["task"] == task:
                by_p.setdefault(row["precision"], []).append(row)
        item_ids, answers = merge_by_item(by_p, value_field="extracted_answer")
        changed = [
            [answers[a][i] != answers[b][i] for i in range(len(item_ids))] for a, b in PAIRS
        ]
        columns = [i for i in range(len(item_ids)) if any(pair[i] for pair in changed)]
        grid = [[1.0 if pair[i] else 0.0 for i in columns] for pair in changed]
        cmap = matplotlib.colors.ListedColormap([SAME_CELL, CHANGED_CELL])
        ax.pcolormesh(grid, cmap=cmap, edgecolors=SURFACE, linewidth=2, vmin=0, vmax=1)
        ax.invert_yaxis()
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_yticks([0.5, 1.5, 2.5], [f"{a} vs {b}" for a, b in PAIRS], fontsize=9, color=INK_2)
        ax.set_xticks([])
        ax.tick_params(length=0)
        counts = ", ".join(
            f"{a} vs {b}: {sum(p)}" for (a, b), p in zip(PAIRS, changed, strict=True)
        )
        ax.set_title(
            f"{TASK_TITLES[task]}: {len(columns)} items change answer in at least one pair "
            f"({counts})",
            loc="left",
            fontsize=10,
            color=INK,
        )
    fig.legend(
        handles=[
            matplotlib.patches.Patch(color=CHANGED_CELL, label="answer differs"),
            matplotlib.patches.Patch(color=SAME_CELL, label="same answer"),
        ],
        loc="lower right",
        ncol=2,
        frameon=False,
        fontsize=9,
        labelcolor=INK_2,
    )
    fig.suptitle(
        "Which items change their answer between precisions (items that never change omitted)",
        x=0.01,
        ha="left",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def user_text(prompt: str) -> str:
    """Strip the Llama 3.2 template tokens, leaving the user's message."""
    return prompt.split("<|end_header_id|>\n\n", 1)[1].split("<|eot_id|>", 1)[0]


def disagreement_review(capability_rows: list[dict], raw_dir: Path, seed: int = 42) -> list[dict]:
    """Sample FP16-vs-Q4 answer changes for manual categorisation (Appendix A).

    Written as a CSV so it opens in Excel; the `category` column is filled by hand.

    Draws up to 8 GSM8K, 8 ARC and 6 behavioural refusal flips. Behavioural
    text comes from the local unredacted files; without them those rows are skipped.
    """
    rng = random.Random(seed)
    review = []
    for task, quota in (("gsm8k", 8), ("arc_challenge", 8)):
        f16 = {
            r["item_id"]: r
            for r in capability_rows
            if r["task"] == task and r["precision"] == "f16"
        }
        q4 = {
            r["item_id"]: r
            for r in capability_rows
            if r["task"] == task and r["precision"] == "q4"
        }
        flips = sorted(
            i for i in f16 if i in q4 and f16[i]["extracted_answer"] != q4[i]["extracted_answer"]
        )
        for item in sorted(rng.sample(flips, min(quota, len(flips)))):
            review.append(
                {
                    "source": task,
                    "item_id": item,
                    "prompt_text": user_text(f16[item]["prompt_text"]),
                    "ground_truth": f16[item]["ground_truth"],
                    "f16_answer": f16[item]["extracted_answer"],
                    "q4_answer": q4[item]["extracted_answer"],
                    "f16_text": f16[item]["generated_text"],
                    "q4_text": q4[item]["generated_text"],
                    "category": "",
                }
            )
    unredacted = [
        row
        for p in sorted(raw_dir.glob("behavioural_*_unredacted.jsonl"))
        for row in read_jsonl(p)
    ]
    f16 = {(r["set"], r["item_id"]): r for r in unredacted if r["precision"] == "f16"}
    q4 = {(r["set"], r["item_id"]): r for r in unredacted if r["precision"] == "q4"}
    flips = sorted(
        k
        for k in f16
        if k in q4
        and f16[k]["refusal_classification_corrected"] != q4[k]["refusal_classification_corrected"]
    )
    for key in sorted(rng.sample(flips, min(6, len(flips)))):
        review.append(
            {
                "source": f"behavioural_{key[0]}",
                "item_id": key[1],
                "prompt_text": user_text(f16[key]["prompt_text"]),
                "ground_truth": None,
                "f16_answer": f16[key]["refusal_classification_corrected"],
                "q4_answer": q4[key]["refusal_classification_corrected"],
                "f16_text": f16[key]["generated_text"],
                "q4_text": q4[key]["generated_text"],
                "category": "",
            }
        )
    return review


def main(
    raw_dir: Path = RAW_DIR, summary_dir: Path = SUMMARY_DIR, figures_dir: Path = FIGURES_DIR
) -> None:
    """Write every summary and figure that the available raw results support."""
    summary_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    systems = load_rows("systems_*.jsonl", raw_dir)
    capability_rows = load_rows("capability_*.jsonl", raw_dir)
    reliability = load_rows("reliability_*.jsonl", raw_dir)
    behavioural = load_rows("behavioural_*.jsonl", raw_dir)

    written = []
    capability = None
    if systems:
        systems_summary(systems).to_csv(summary_dir / "systems_summary.csv", index=False)
        written.append("systems_summary.csv")
    if capability_rows:
        capability = capability_summary(capability_rows)
        capability.to_csv(summary_dir / "capability_summary.csv", index=False)
        agreement_summary(capability_rows).to_csv(
            summary_dir / "agreement_summary.csv", index=False
        )
        written += ["capability_summary.csv", "agreement_summary.csv"]
        figure_disagreement(capability_rows, figures_dir / "fig2_disagreement.png")
        written.append("fig2_disagreement.png")
        if systems:
            figure_pareto(systems, capability, figures_dir / "fig1_pareto.png")
            written.append("fig1_pareto.png")
    if reliability:
        reliability_summary(reliability).to_csv(
            summary_dir / "reliability_summary.csv", index=False
        )
        written.append("reliability_summary.csv")
    # Both classifiers are reported (PROTOCOL.md deviation 12); no suffix is the frozen one.
    labels = (("", "refusal_classification"), ("_corrected", "refusal_classification_corrected"))
    for suffix, field in labels if behavioural else ():
        beh = behavioural_summary(behavioural, label_field=field)
        beh.to_csv(summary_dir / f"behavioural_summary{suffix}.csv", index=False)
        written.append(f"behavioural_summary{suffix}.csv")
        if capability is not None:
            preregistered_claim(capability, beh).to_csv(
                summary_dir / f"preregistered_claim{suffix}.csv", index=False
            )
            written.append(f"preregistered_claim{suffix}.csv")
    followup = load_rows("followup_*.jsonl", raw_dir)
    if followup:
        behavioural_summary(followup, label_field="refusal_classification_corrected").to_csv(
            summary_dir / "followup_jbb_summary.csv", index=False
        )
        written.append("followup_jbb_summary.csv")
    # Never regenerate the review file once it exists: it holds hand-entered categories.
    review_path = summary_dir / "disagreement_review_unredacted.csv"
    if capability_rows and review_path.exists():
        written.append(f"{review_path.name} kept (hand-filled, never overwritten)")
    elif capability_rows:
        review = disagreement_review(capability_rows, raw_dir)
        path = review_path
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(review[0]))
            writer.writeheader()
            writer.writerows(review)
        written.append(f"{path.name} ({len(review)} rows)")

    for name in written:
        print(f"wrote {name}")
    for suite, rows in (("reliability", reliability), ("behavioural", behavioural)):
        if not rows:
            print(f"skipped {suite}: no raw files yet")


if __name__ == "__main__":
    main()
