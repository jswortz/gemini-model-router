"""
Render the cost/quality frontier from experiments/results_v2/report.json.

Replaces the failed PaperBanana run (gemini-3.1 image API rejected the API key).
matplotlib is the right tool here anyway: this is a scatter plot, not a diagram.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

REPORT = Path(__file__).resolve().parents[1] / "results_v2" / "report.json"
OUT = Path(__file__).resolve().parent / "fig1_cost_quality_frontier.png"


def main() -> None:
    rep = json.loads(REPORT.read_text())
    variants = rep["variants"]

    points = []
    for name, v in variants.items():
        all_block = v["all"]
        cost_lo, cost_mid, cost_hi = all_block["cost_usd_per_prompt"]
        q_lo, q_mid, q_hi = all_block["quality_fit_mean"]
        points.append({
            "name": name,
            "cost": cost_mid * 1000,
            "cost_lo": cost_lo * 1000,
            "cost_hi": cost_hi * 1000,
            "q": q_mid,
            "q_lo": q_lo,
            "q_hi": q_hi,
        })

    fig, ax = plt.subplots(figsize=(10, 7), dpi=130)

    # Style by variant family
    colors = {
        "router": "#1f77b4",  # blue
        "naive":  "#d62728",  # red
    }
    naive = {"all_gemma4", "all_gemini", "all_claude"}

    for p in points:
        is_naive = p["name"] in naive
        c = colors["naive"] if is_naive else colors["router"]
        marker = "s" if is_naive else "o"
        ax.errorbar(
            p["cost"], p["q"],
            xerr=[[max(0.0, p["cost"] - p["cost_lo"])], [max(0.0, p["cost_hi"] - p["cost"])]],
            yerr=[[max(0.0, p["q"] - p["q_lo"])], [max(0.0, p["q_hi"] - p["q"])]],
            fmt=marker, color=c, ecolor=c, alpha=0.8,
            markersize=10, capsize=3,
            label=None,
        )
        # Label placement: nudge to avoid overlap
        dx, dy = 0.02, 0.012
        if p["name"] == "all_gemma4":
            dx, dy = 0.02, 0.025
        if p["name"] == "all_claude":
            dx, dy = -0.05, -0.04
        if p["name"] == "all_gemini":
            dx, dy = 0.02, -0.025
        ax.annotate(p["name"], xy=(p["cost"], p["q"]),
                    xytext=(p["cost"] + dx, p["q"] + dy),
                    fontsize=9, color=c)

    # Pareto frontier (router-only): lowest cost at each quality threshold (descending)
    router_pts = [p for p in points if p["name"] not in naive]
    router_pts.sort(key=lambda p: p["q"], reverse=True)
    pareto, best = [], float("inf")
    for p in router_pts:
        if p["cost"] < best:
            pareto.append(p)
            best = p["cost"]
    px = [p["cost"] for p in pareto]
    py = [p["q"] for p in pareto]
    ax.plot(px, py, ":", color=colors["router"], alpha=0.5, label="router Pareto frontier")

    # Annotated callouts for the headline finding
    ax.annotate(
        "router beats all_claude:\n−35% cost, +25% qfit\n(p=0.010, n=1162)",
        xy=(points[0]["cost"], points[0]["q"]),  # baseline
        xytext=(0.45, 0.50),
        fontsize=10, color="black",
        bbox=dict(boxstyle="round,pad=0.4", fc="#ffffe0", ec="gray"),
        arrowprops=dict(arrowstyle="->", color="gray", alpha=0.6),
    )

    # Projected specialist (synthetic, illustrative)
    ax.scatter([0.0], [0.85], marker="*", s=300, color="#2ca02c", zorder=5,
               label="projected (Gemma+LoRA specialist, ≈$0)")
    ax.annotate("gemma4:cloudrun\n(projected; SLM_PLAN.md)",
                xy=(0.0, 0.85), xytext=(0.05, 0.88),
                fontsize=9, color="#2ca02c")

    ax.set_xlabel("cost per request (USD × 10⁻³)  ←lower is better", fontsize=11)
    ax.set_ylabel("quality_fit (MiniLM cosine to chosen-backend centroid)  ↑higher is better", fontsize=11)
    ax.set_title("Router replay: cost vs. quality_fit across 11 configs (n=1162; bootstrap 95% CI)\n"
                 "Naive single-backend baselines vs. router variants", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.4)

    # Legend handles
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["router"],
               markersize=10, label="router variants"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=colors["naive"],
               markersize=10, label="naive single-backend"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#2ca02c",
               markersize=15, label="projected fine-tuned specialist"),
        Line2D([0], [0], linestyle=":", color=colors["router"], label="router Pareto frontier"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
