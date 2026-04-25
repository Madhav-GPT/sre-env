# Eval results

This directory holds artifacts produced by [`notebooks/02_basic_eval_comparison.ipynb`](../../notebooks/02_basic_eval_comparison.ipynb):

| File | What it contains |
|---|---|
| `comparison_raw.csv` | One row per (policy, scenario, episode) eval episode |
| `comparison_summary.csv` | Per-policy aggregates (mean / median / p25 / p75 / resolved-rate) |
| `comparison_table.csv` | Printable table for the README |
| `comparison_per_template.png` | Per-template box-and-whisker reward distributions |
| `comparison_hero.png` | Single-axis bar chart — the README hero figure |

These files are produced by running the eval notebook against:

- the held-out 12-scenario set (one `__p05` procgen variant per template)
- 7 policies: random, heuristic, scripted-optimal, Llama-3.3-70B, Claude Haiku, Claude Sonnet, trained Qwen2.5-3B
- 3 episodes per (policy, scenario)

= 252 evaluation episodes per full sweep.

The `.gitignore` excludes the artifact files themselves (so this directory stays small until a real run produces them); only this README and `.gitkeep` are committed.

To run:

```bash
make install-train
jupyter nbconvert --to notebook --execute notebooks/02_basic_eval_comparison.ipynb
```

Or run the notebook interactively in Colab.
