"""Build the technical note's markdown tables (spec section 18) from the summaries.

Reads results/summary/*.csv (written by 05_aggregate_and_plot.py) and the raw
systems rows, and writes results/summary/report_tables.md. A table is skipped if
its input does not exist yet.
"""

import json
from pathlib import Path

import pandas as pd

RAW_DIR = Path("results/raw")
SUMMARY_DIR = Path("results/summary")
PRECISIONS = ("f16", "q8", "q4")
TASK_NAMES = {"gsm8k": "GSM8K", "arc_challenge": "ARC-Challenge"}


def pct(value: float) -> str:
    """Format a proportion as a percentage."""
    return f"{100 * value:.1f}%"


def pct_ci(value: float, low: float, high: float) -> str:
    """Format a proportion with its 95% CI."""
    return f"{100 * value:.1f}% [{100 * low:.1f}, {100 * high:.1f}]"


def points_ci(value: float, low: float, high: float) -> str:
    """Format a difference of proportions in percentage points with its CI."""
    return f"{100 * value:+.1f} pts [{100 * low:+.1f}, {100 * high:+.1f}]"


def table(header: list[str], rows: list[list[str]]) -> str:
    """Render a markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def systems_table(raw_dir: Path) -> str | None:
    """Table 1: per-precision systems figures over every measured generation."""
    rows = [
        json.loads(line)
        for path in sorted(raw_dir.glob("systems_*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    if not rows:
        return None
    df = pd.DataFrame(rows)
    out = []
    for precision in PRECISIONS:
        p = df[df["precision"] == precision]
        out.append(
            [
                precision,
                f"{p['load_time_s'].mean():.2f}",
                f"{p['ttft_ms'].quantile(0.50):.1f}",
                f"{p['ttft_ms'].quantile(0.95):.1f}",
                f"{p['decode_tok_s'].mean():.1f}",
                f"{p['peak_vram_mb'].mean():,.0f}",
                f"{p['peak_rss_mb'].mean():,.0f}",
            ]
        )
    header = [
        "Precision",
        "Load (s)",
        "TTFT p50 (ms)",
        "TTFT p95 (ms)",
        "Tok/s",
        "Peak VRAM (MB)",
        "Peak RSS (MB)",
    ]
    return table(header, out)


def capability_table(cap: pd.DataFrame) -> str:
    """Table 2: accuracy per precision and the paired FP16-Q4 delta."""
    out = []
    for task, name in TASK_NAMES.items():
        t = cap[cap["task"] == task].set_index("precision")
        if t.empty:
            continue
        cells = [
            pct_ci(t.loc[p, "mean_score"], t.loc[p, "ci_lower"], t.loc[p, "ci_upper"])
            for p in PRECISIONS
        ]
        q4 = t.loc["q4"]
        out.append(
            [
                name,
                str(int(t.loc["f16", "n_items"])),
                *cells,
                points_ci(q4["delta_vs_f16"], q4["delta_ci_lower"], q4["delta_ci_upper"]),
                f"{q4['cohens_d_vs_f16']:.2f}",
            ]
        )
    header = ["Task", "N", "FP16 [CI]", "Q8 [CI]", "Q4 [CI]", "Delta FP16-Q4 [CI]", "Cohen's d"]
    return table(header, out)


def agreement_table(cap: pd.DataFrame, agree: pd.DataFrame) -> str:
    """Table 3: answer-level agreement FP16-Q4 and the score-level disagreement."""
    out = []
    for task, name in TASK_NAMES.items():
        a = agree[(agree["task"] == task) & (agree["precision"] == "q4")]
        c = cap[(cap["task"] == task) & (cap["precision"] == "q4")]
        if a.empty:
            continue
        a, c = a.iloc[0], c.iloc[0]
        out.append(
            [
                name,
                pct(1 - a["answer_flip_rate_vs_f16"]),
                pct_ci(a["answer_flip_rate_vs_f16"], a["flip_ci_lower"], a["flip_ci_upper"]),
                pct(c["disagreement_rate_vs_f16"]),
                f"{int(a['items_flipped_vs_f16'])} of {int(a['n_items'])}",
            ]
        )
    header = [
        "Task",
        "Exact answer agreement F16-Q4",
        "Answer flip rate [CI]",
        "Correctness disagreement",
        "Items flipped",
    ]
    return table(header, out)


def reliability_table(rel: pd.DataFrame) -> str:
    """Table 4: self-consistency across seeds per precision."""
    out = [
        [
            r["precision"],
            pct_ci(
                r["self_consistency_rate"],
                r["self_consistency_ci_lower"],
                r["self_consistency_ci_upper"],
            ),
            pct(r["texts_differ_rate"]),
        ]
        for _, r in rel.iterrows()
    ]
    return table(["Precision", "Self-consistency rate [CI]", "Items whose 3 texts differ"], out)


def behavioural_table(beh: pd.DataFrame) -> str:
    """Table 5: refusal and attack success per precision, with change vs FP16."""
    out = []
    for set_name, label in (("harmful", "Harmful"), ("jailbreak", "Jailbreak")):
        s = beh[beh["set"] == set_name].set_index("precision")
        if s.empty:
            continue
        q4 = s.loc["q4"]
        out.append(
            [
                label,
                "Refusal rate",
                *(pct(s.loc[p, "refusal_rate"]) for p in PRECISIONS),
                points_ci(
                    q4["refusal_delta_vs_f16"],
                    q4["refusal_delta_ci_lower"],
                    q4["refusal_delta_ci_upper"],
                ),
            ]
        )
        if set_name == "jailbreak":
            out.append(
                [
                    label,
                    "Attack success rate",
                    *(pct(s.loc[p, "attack_success_rate"]) for p in PRECISIONS),
                    points_ci(
                        -q4["refusal_delta_vs_f16"],
                        -q4["refusal_delta_ci_upper"],
                        -q4["refusal_delta_ci_lower"],
                    ),
                ]
            )
        out.append(
            [
                label,
                "Refusal flip rate vs FP16",
                "",
                pct_ci(
                    s.loc["q8", "flip_rate_vs_f16"],
                    s.loc["q8", "flip_ci_lower"],
                    s.loc["q8", "flip_ci_upper"],
                ),
                pct_ci(q4["flip_rate_vs_f16"], q4["flip_ci_lower"], q4["flip_ci_upper"]),
                "",
            ]
        )
    return table(["Set", "Metric", "FP16", "Q8", "Q4", "Delta Q4 vs FP16 [CI]"], out)


def claim_table(claim: pd.DataFrame) -> str:
    """The pre-registered claim: movements FP16 to Q4 and the verdict."""
    out = []
    for _, r in claim.iterrows():
        if r["metric"] == "claim":
            continue
        rel = "undefined" if pd.isna(r["relative_movement"]) else pct(r["relative_movement"])
        out.append(
            [
                r["metric"],
                pct(r["f16"]),
                pct(r["q4"]),
                f"{100 * r['absolute_movement']:.1f} pts",
                rel,
            ]
        )
    verdict = claim[claim["metric"] == "claim"]["verdict"].iloc[0]
    body = table(["Metric", "FP16", "Q4", "Absolute movement", "Relative movement"], out)
    return f"{body}\n\n**Pre-registered claim: {verdict}.**"


def followup_table(df: pd.DataFrame) -> str:
    """Table F1: refusal behaviour on the JailbreakBench follow-up sets."""
    out = []
    names = (
        ("jbb_harmful", "JBB harmful: refusal rate"),
        ("jbb_benign", "JBB benign: over-refusal rate"),
        ("jbb_pair", "PAIR attacks: refusal rate"),
    )
    for set_name, label in names:
        s = df[df["set"] == set_name].set_index("precision")
        if s.empty:
            continue
        q4 = s.loc["q4"]
        out.append(
            [
                label,
                str(int(s.loc["f16", "n_items"])),
                *(
                    pct_ci(
                        s.loc[p, "refusal_rate"],
                        s.loc[p, "refusal_ci_lower"],
                        s.loc[p, "refusal_ci_upper"],
                    )
                    for p in PRECISIONS
                ),
                points_ci(
                    q4["refusal_delta_vs_f16"],
                    q4["refusal_delta_ci_lower"],
                    q4["refusal_delta_ci_upper"],
                ),
                pct_ci(q4["flip_rate_vs_f16"], q4["flip_ci_lower"], q4["flip_ci_upper"]),
            ]
        )
    header = [
        "Set",
        "N",
        "FP16 [CI]",
        "Q8 [CI]",
        "Q4 [CI]",
        "Delta Q4 vs FP16 [CI]",
        "Flip rate Q4 [CI]",
    ]
    return table(header, out)


def main(raw_dir: Path = RAW_DIR, summary_dir: Path = SUMMARY_DIR) -> None:
    """Write report_tables.md with every table the available summaries support."""

    def csv(name: str) -> pd.DataFrame | None:
        path = summary_dir / name
        return pd.read_csv(path) if path.exists() else None

    cap, agree = csv("capability_summary.csv"), csv("agreement_summary.csv")
    rel = csv("reliability_summary.csv")
    f1 = csv("followup_jbb_summary.csv")
    beh, beh_frozen = csv("behavioural_summary_corrected.csv"), csv("behavioural_summary.csv")
    claim, claim_frozen = csv("preregistered_claim_corrected.csv"), csv("preregistered_claim.csv")
    sections = [
        ("Table 1: Systems", systems_table(raw_dir)),
        ("Table 2: Capability", capability_table(cap) if cap is not None else None),
        (
            "Table 3: Agreement (FP16 vs Q4)",
            agreement_table(cap, agree) if cap is not None and agree is not None else None,
        ),
        ("Table 4: Reliability", reliability_table(rel) if rel is not None else None),
        (
            "Table 5: Behavioural (corrected classifier)",
            behavioural_table(beh) if beh is not None else None,
        ),
        (
            "Table 5a: Behavioural (frozen classifier, as originally scored)",
            behavioural_table(beh_frozen) if beh_frozen is not None else None,
        ),
        (
            "Pre-registered claim (FP16 vs Q4), corrected classifier",
            claim_table(claim) if claim is not None else None,
        ),
        (
            "Pre-registered claim (FP16 vs Q4), frozen classifier",
            claim_table(claim_frozen) if claim_frozen is not None else None,
        ),
        (
            "Table F1: Follow-up, JailbreakBench (exploratory, corrected classifier)",
            followup_table(f1) if f1 is not None else None,
        ),
    ]
    parts = ["# Report tables", "", "Generated by `scripts/06_build_report_tables.py`.", ""]
    for title, body in sections:
        if body is None:
            print(f"skipped {title}: input not available yet")
            continue
        parts += [f"## {title}", "", body, ""]
    out = summary_dir / "report_tables.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
