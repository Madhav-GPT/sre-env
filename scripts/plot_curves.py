"""Plot training-curve PNGs from JSONL trajectories.

Usage:
    python scripts/plot_curves.py train/data/eval_sweep.jsonl --output eval/results/training_curve.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=str, help="JSONL file with one row per episode/step.")
    parser.add_argument("--output", type=str, default="eval/results/curve.png")
    parser.add_argument("--x-field", default="step", help="Field on the x-axis (default: step).")
    parser.add_argument("--y-field", default="final_score",
                        help="Field on the y-axis (default: final_score).")
    parser.add_argument("--group-by", default=None, help="Optional grouping field (e.g. policy).")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    if not rows:
        print("Empty input — nothing to plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    if args.group_by:
        groups: dict[str, list[tuple[float, float]]] = {}
        for r in rows:
            key = str(r.get(args.group_by, "default"))
            groups.setdefault(key, []).append((r.get(args.x_field, 0), r.get(args.y_field, 0)))
        for key, pairs in groups.items():
            xs, ys = zip(*sorted(pairs))
            ax.plot(xs, ys, label=key, marker=".", linewidth=1.5, alpha=0.8)
        ax.legend()
    else:
        xs = [r.get(args.x_field, i) for i, r in enumerate(rows)]
        ys = [r.get(args.y_field, 0) for r in rows]
        ax.plot(xs, ys, marker=".", linewidth=1.5)

    ax.set_xlabel(args.x_field)
    ax.set_ylabel(args.y_field)
    ax.set_title(args.title or f"{args.y_field} over {args.x_field}")
    ax.grid(alpha=0.3)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
