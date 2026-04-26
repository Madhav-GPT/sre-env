# Eval results

Artifacts produced by cell 10/11 of [`notebooks/01_triage_train_grpo_qwen25_7b.ipynb`](../../notebooks/01_triage_train_grpo_qwen25_7b.ipynb) and the standalone comparison notebook [`notebooks/02_triage_eval_compare_all.ipynb`](../../notebooks/02_triage_eval_compare_all.ipynb):

| File | What it contains |
|---|---|
| `qwen25_7b_comparison_raw.csv` | One row per (policy, scenario, seed) episode |
| `qwen25_7b_comparison_summary.csv` | Per-policy aggregates (mean / median / p25 / p75 / resolved-rate) |
| `qwen25_7b_comparison_hero.png` | Single-axis bar chart with whiskers — the README hero figure |
| `qwen25_7b_comparison_per_template.png` | Per-template grouped bars by policy |

The full sweep covers:

- The held-out 12-scenario set (one `__p05` procgen variant per template)
- Up to 5 policies: `random`, `heuristic`, `scripted_optimal`, `qwen25-7b-sft-only`, `qwen25-7b-grpo`
- 3 seeds per (policy, scenario)
- = 180 evaluation episodes per full sweep

## Latest measured numbers (Qwen2.5-7B, A100 80GB)

| policy | mean | median | p25 | p75 | resolved_rate |
|---|---|---|---|---|---|
| random | 0.342 | 0.378 | 0.340 | 0.380 | 0/36 |
| qwen25-7b-sft-only | 0.379 | 0.380 | 0.378 | 0.380 | 0/36 |
| qwen25-7b-grpo | 0.379 | 0.380 | 0.378 | 0.380 | 0/36 |
| heuristic | 0.704 | 0.705 | 0.703 | 0.705 | 0/36 |
| scripted-optimal | 0.938 | 0.939 | 0.937 | 0.940 | 36/36 |

SFT lifted the model 11% above random. GRPO did not move the mean further at K=2 / 40-step / 7B budget. Both stay below the heuristic plateau at 0.704 — the rubric refuses partial credit for "looks right" output. See [`README.md`](../../README.md) §"Training & datasets" for the framing.

## To re-run

Open the notebook in Colab / HF Space, set the runtime to A100 80GB, paste an HF token, run-all. Resume points: if `outputs/qwen25_7b_grpo_final/adapter_model.safetensors` exists, training cells skip and eval re-runs.
