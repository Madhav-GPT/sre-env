# Claude seed trajectories

**Provenance.** These trajectories were generated on 2026-04-24 by Claude
Opus 4.7 (1M context) driving the live sre-gym env through the OpenClaw
pool server (`openclaw_integration/pool_server.py`), step by step. No
runbook knowledge was fed in — every action was chosen from the observation
alone.

**Files.**
- `claude_<scenario_id>.jsonl` — raw event logs (reset / step / evaluate
  events). One file per episode. Small enough to read and audit by hand.
- `claude_seed.jsonl` — canonical training format. Produced by
  `train/compile_claude_seed.py`. Schema matches the output of
  `train/collect_trajectories.py`, so SFT/GRPO pipelines treat these
  identically to API-driven trajectories.

**Stats (6 episodes, all resolved, mean score 0.769):**

| scenario | score | steps |
|---|---|---|
| worker_deploy_cascade | 0.773 | 7 |
| worker_deploy_cascade__p02 | 0.773 | 7 |
| db_config_rollout | 0.785 | 7 |
| db_config_rollout__p01 | 0.785 | 7 |
| gateway_auth_rollout | 0.714 | 5 |
| gateway_auth_rollout__p03 | 0.781 | 6 |

**Notes.**
- Hard scenario (0.714 vs 0.781 on the variant): I deliberately skipped
  `database_recovery` on the first hard run to probe whether the grader
  rewards completeness over speed. It does — running both checks on the
  variant earned the verification points back and still beat the scripted
  baseline via speed_bonus. Both trajectories are useful training data:
  one shows the trade-off going the wrong way, one shows it going the
  right way.

**Regenerate.** Boot the pool server and re-run the steps in
`/tmp/play.py` via the session the driver conversation used. The raw
event files are deterministic given scenario_id + same action sequence.
