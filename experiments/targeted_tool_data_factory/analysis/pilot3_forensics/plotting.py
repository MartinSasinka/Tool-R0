"""Optional decision plots (CSV always primary)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=140)


def try_plots(out_dir: Path, ctx: Dict[str, Any]) -> List[str]:
    written: List[str] = []
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return written

    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    headline = ctx.get("headline") or {}
    # 1. paired transitions
    labels = ["win→win", "loss→loss", "loss→win", "win→loss"]
    vals = [
        headline.get("win_to_win", 0),
        headline.get("loss_to_loss", 0),
        headline.get("loss_to_win", 0),
        headline.get("win_to_loss", 0),
    ]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(labels, vals, color=["#2a9d8f", "#6c757d", "#2b8a3e", "#c92a2a"])
    ax.set_title("C0 vs D1 paired transitions")
    ax.set_ylabel("count")
    _save(fig, plots / "paired_transitions.png")
    plt.close(fig)
    written.append("plots/paired_transitions.png")

    # 2. win by bucket
    buckets = headline.get("by_call_bucket") or {}
    keys = [k for k in ["2", "3", "4", "5", "6+"] if k in buckets and buckets[k].get("n")]
    if keys:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        x = range(len(keys))
        ax.bar([i - 0.2 for i in x], [100 * buckets[k]["win_rate_c0"] for k in keys], width=0.4, label="C0")
        ax.bar([i + 0.2 for i in x], [100 * buckets[k]["win_rate_d1"] for k in keys], width=0.4, label="D1")
        ax.set_xticks(list(x))
        ax.set_xticklabels(keys)
        ax.set_ylabel("win %")
        ax.set_title("Win rate by call-count bucket")
        ax.legend()
        _save(fig, plots / "win_by_call_bucket.png")
        plt.close(fig)
        written.append("plots/win_by_call_bucket.png")

    # 3. failure transition heatmap (top categories)
    matrix = ctx.get("failure_matrix") or []
    if matrix:
        from collections import Counter
        marg: Counter = Counter()
        for r in matrix:
            marg[r["c0_primary"]] += int(r["count"])
            marg[r["d1_primary"]] += int(r["count"])
        labels_m = [k for k, _ in marg.most_common(8)]
        idx = {k: i for i, k in enumerate(labels_m)}
        nlab = len(labels_m)
        M = [[0.0 for _ in range(nlab)] for _ in range(nlab)]
        for r in matrix:
            a, b = r["c0_primary"], r["d1_primary"]
            if a in idx and b in idx:
                M[idx[a]][idx[b]] += float(r["count"])
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(M, cmap="Blues")
        ax.set_xticks(range(nlab))
        ax.set_yticks(range(nlab))
        ax.set_xticklabels(labels_m, rotation=90, fontsize=7)
        ax.set_yticklabels(labels_m, fontsize=7)
        ax.set_xlabel("D1 primary")
        ax.set_ylabel("C0 primary")
        ax.set_title("Failure transition heatmap (top labels)")
        fig.colorbar(im, ax=ax, fraction=0.046)
        _save(fig, plots / "failure_transition_heatmap.png")
        plt.close(fig)
        written.append("plots/failure_transition_heatmap.png")

    # 4/5 topology freq + cumulative
    topo_dist = ctx.get("topology_distribution_rows") or []
    if topo_dist:
        train = [r for r in topo_dist if r.get("source") == "train300"][:30]
        if train:
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.bar(range(len(train)), [r["count"] for r in train])
            ax.set_title("Train-300 topology frequency (top 30)")
            ax.set_xlabel("rank")
            ax.set_ylabel("count")
            _save(fig, plots / "topology_freq_train300.png")
            plt.close(fig)
            written.append("plots/topology_freq_train300.png")
            total = sum(r["count"] for r in train) or 1
            cum = []
            s = 0
            # use full counter if provided
            full = ctx.get("topology_train300_full_counts") or [r["count"] for r in train]
            for c in full:
                s += c
                cum.append(s / sum(full))
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(range(1, len(cum) + 1), cum)
            ax.set_title("Topology cumulative concentration (train300)")
            ax.set_xlabel("n topologies")
            ax.set_ylabel("cumulative share")
            _save(fig, plots / "topology_cumulative_train300.png")
            plt.close(fig)
            written.append("plots/topology_cumulative_train300.png")

    # 6 coverage vs outcome
    cov = ctx.get("coverage_by_outcome") or []
    cov_exact = [r for r in cov if str(r.get("bucket", "")).startswith("exact")]
    if cov_exact:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar([r["bucket"] for r in cov_exact], [r.get("net_gain", 0) for r in cov_exact])
        ax.set_title("Coverage bucket vs net D1−C0 gain")
        ax.tick_params(axis="x", rotation=30)
        _save(fig, plots / "coverage_vs_net_gain.png")
        plt.close(fig)
        written.append("plots/coverage_vs_net_gain.png")

    # 7 OOD decile
    ood_dec = ((ctx.get("distribution") or {}).get("summary") or {}).get("by_ood_decile") or []
    if ood_dec:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        xs = [r["decile"] for r in ood_dec if r.get("n")]
        ax.plot(xs, [100 * (r["c0_win_rate"] or 0) for r in ood_dec if r.get("n")], label="C0")
        ax.plot(xs, [100 * (r["d1_win_rate"] or 0) for r in ood_dec if r.get("n")], label="D1")
        ax.set_title("OOD decile vs win rate")
        ax.set_xlabel("OOD decile")
        ax.set_ylabel("win %")
        ax.legend()
        _save(fig, plots / "ood_decile_vs_win.png")
        plt.close(fig)
        written.append("plots/ood_decile_vs_win.png")

    # 8 train300 vs rest — generation cell TV already in tables; plot top share diffs
    subset_rows = ctx.get("subset_rows") or []
    cell_rows = [r for r in subset_rows if r.get("feature") == "generation_cell"][:20]
    if cell_rows:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.bar(range(len(cell_rows)), [r.get("share_diff") or 0 for r in cell_rows])
        ax.set_title("Train300 − rest300 generation_cell share diff (top rows)")
        _save(fig, plots / "train300_vs_rest_cell_diff.png")
        plt.close(fig)
        written.append("plots/train300_vs_rest_cell_diff.png")

    # 9 dead by cell if available
    by_cell = (ctx.get("reward") or {}).get("groups_by_cell") or []
    if by_cell:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        top = sorted(by_cell, key=lambda r: -r.get("n_groups", 0))[:20]
        ax.bar(range(len(top)), [r.get("dead_rate") or 0 for r in top])
        ax.set_title("Dead-group rate by generation cell (if rollouts)")
        _save(fig, plots / "dead_rate_by_cell.png")
        plt.close(fig)
        written.append("plots/dead_rate_by_cell.png")

    # 10 distractor hardness
    dh = (ctx.get("surface") or {}).get("distractor") or {}
    if dh.get("train_mean") is not None and dh.get("diag_mean") is not None:
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.bar(["train", "diagnostic"], [dh["train_mean"], dh["diag_mean"]])
        ax.set_title("Mean distractor hardness proxy")
        _save(fig, plots / "distractor_hardness.png")
        plt.close(fig)
        written.append("plots/distractor_hardness.png")

    return written
