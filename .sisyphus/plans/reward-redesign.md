# Reward Redesign for Unified Incident Env

## TL;DR
> **Summary**: Replace breadcrumb-based rewards with a world-state-based reward system: normalized step cost, incident-health delta shaping, a tiny non-farmable hypothesis-quality bonus, and terminal bonuses/penalties tied to verified containment and recovery. Keep the public deterministic benchmark score separate from training reward, but remove breadcrumb terms from both.
> **Deliverables**:
> - Reworked training-time step reward in `unified_incident_env/server/environment.py`
> - Reworked public deterministic score in `unified_incident_env/server/grader.py`
> - Structured hypothesis payload on `classify_vulnerability`
> - Scenario-authored critical-path service weights and reward config in `server/challenge.py`
> - Updated prompts/inference/tests for the structured hypothesis contract
> - Regression tests proving breadcrumb rewards are gone and world-improving actions dominate
> **Effort**: Large
> **Parallel**: YES - 4 waves
> **Critical Path**: Task 1 → Task 2 → Task 3 → Task 6

## Context
### Original Request
Redesign the reward system so points come from world improvement, step cost, and small calibrated hypothesis quality rather than from the environment revealing the “correct branch.” Keep the env compatible with Gymnasium/OpenEnv-style RL where every `step(action)` returns reward, but tie reward to state-transition quality rather than clue clicks.

### Interview Summary
- Reward must still be emitted on every step.
- Investigation actions should mostly cost time and should not directly reward clue discovery.
- Hypothesis actions can receive a small score for decision quality: root-cause accuracy, service localization, confidence calibration, and recommended next action quality.
- Big rewards should remain tied to actual containment, verified recovery, and correct final resolution.
- Reward shaping should follow the spirit of potential-based shaping: dense guidance via better state, not better clue collection.
- Training can run on Colab/Kaggle; environment logic remains local.

### Metis Review (gaps addressed)
- Added a strict **reward whitelist** and **forbidden-source blacklist**.
- Made hypothesis reward explicitly one-time and non-farmable.
- Separated training reward from public deterministic benchmark score.
- Normalized step costs by scenario budget to avoid punishing longer scenarios unfairly.
- Added explicit regression checks for reward/public-score drift.
- Resolved hidden ambiguity: reuse `classify_vulnerability` instead of introducing a new `submit_hypothesis` action.

## Work Objectives
### Core Objective
Refactor the benchmark so the agent learns from state improvement and decision quality, not from authored breadcrumb rewards, while preserving a deterministic public evaluation contract.

### Deliverables
- `server/environment.py` returns step rewards based on:
  - normalized step cost
  - delta incident-health potential
  - one-time hypothesis bonus/penalty
  - terminal outcome bonus/penalty
  - explicit unsafe/redundant action penalties
- `server/grader.py` computes public `final_score` without rewarding evidence discovery, patch-id guessing, or stage progression by itself.
- `server/challenge.py` contains per-scenario critical-path service weights and reward-config metadata.
- `models.py` extends `classify_vulnerability` payload to carry hypothesis scoring fields.
- `trainer/prompts.py` and `inference.py` understand the structured hypothesis payload.
- Tests cover reward decomposition, non-farmable hypothesis scoring, and terminal correctness.

### Definition of Done (verifiable conditions with commands)
- `./.venv/bin/pytest unified_incident_env/tests -q` exits 0.
- For a fixed scenario, a pure query action yields only step cost / redundancy effects, not positive breadcrumb reward.
- For a fixed scenario, verified containment/recovery yields positive reward deltas.
- Repeating the same hypothesis does not mint additional bonus.
- Public deterministic score no longer uses `relevant_investigations` or any direct clue-count term.

### Must Have
- No direct positive reward for evidence discovery, unlock events, query success, patch-id selection, or stage advancement.
- Incident-health potential derived only from verified/public world state.
- `classify_vulnerability` supports structured hypothesis scoring with cause, services, confidence, and next action.
- Training reward and public score are both documented and distinguishable.

### Must NOT Have
- No new `submit_hypothesis` action unless the existing `classify_vulnerability` path proves insufficient during implementation review.
- No hidden proxy breadcrumb reward through internal fields like `matched_evidence_ids`, `unlock_threshold`, or `infra_progress`.
- No reward mutation outside the actual returned `reward` from `step()`.
- No acceptance criteria that depend on human eyeballing logs.

## Verification Strategy
> ZERO HUMAN INTERVENTION - all verification is agent-executed.
- Test decision: tests-after with existing `pytest` suite plus new deterministic reward regression tests.
- QA policy: every implementation task includes agent-executed assertions on reward sign/magnitude and action/schema behavior.
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.{ext}`

## Execution Strategy
### Parallel Execution Waves
Wave 1: reward-model foundation and schema decisions
- Task 1: define allowed/forbidden reward sources and scenario reward config
- Task 2: extend action/state schema for structured hypotheses
- Task 3: implement incident-health potential helpers

Wave 2: core scoring rewrite
- Task 4: replace step reward logic in environment
- Task 5: replace public deterministic score breakdown
- Task 6: update scenario metadata and authored weights

Wave 3: contract consumers
- Task 7: update prompts, response schema, and parser expectations
- Task 8: update inference fallback/hypothesis generation
- Task 9: update baseline/walkthrough/tests for new hypothesis payload

Wave 4: regression and training-path hardening
- Task 10: add reward decomposition/regression tests
- Task 11: add reward/public-score drift checks for fixed scenarios
- Task 12: document Colab/Kaggle GRPO usage against the new reward semantics

### Dependency Matrix (full, all tasks)
- Task 1 blocks Tasks 3, 4, 5, 6.
- Task 2 blocks Tasks 7, 8, 9.
- Task 3 blocks Task 4.
- Task 4 blocks Tasks 10 and 11.
- Task 5 blocks Task 11.
- Task 6 blocks Task 4 and Task 5.
- Task 7 blocks Task 8 and Task 9.
- Task 8 blocks Task 12.
- Task 9 blocks Task 10.
- Tasks 10 and 11 block final verification wave.

### Agent Dispatch Summary
- Wave 1 → 3 tasks → deep / oracle-consulted / quick
- Wave 2 → 3 tasks → deep / unspecified-high
- Wave 3 → 3 tasks → quick / unspecified-high
- Wave 4 → 3 tasks → quick / writing / unspecified-high

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [ ] 1. Define reward whitelist, blacklist, and config schema

  **What to do**: Add a single source of truth for reward terms in `server/challenge.py` or a nearby reward-config module. Define which signals are allowed to contribute to training reward and which are forbidden. Add per-scenario `critical_service_weights`, `step_cost_scale`, and hypothesis-bonus constants. Remove authored dependence on clue/evidence counts from the new reward path.
  **Must NOT do**: Do not yet rewrite reward logic in `environment.py`; do not add a new action type.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: this is the architecture lock for all later reward logic.
  - Skills: `[]` - no special skill required.
  - Omitted: `[omarchy]` - unrelated domain.

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 3,4,5,6 | Blocked By: none

  **References**:
  - Pattern: `unified_incident_env/server/challenge.py:96-156,284-345,486-546` - current evidence/unlock/verify metadata to replace or augment.
  - Pattern: `unified_incident_env/server/environment.py:263-323` - current breadcrumb reward path.
  - Pattern: `unified_incident_env/server/grader.py:73-128` - current public score terms.
  - External: `https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf` - shaping must preserve the right objective.
  - External: `https://github.com/Farama-Foundation/Gymnasium` - step rewards should reflect environment transition quality.

  **Acceptance Criteria**:
  - [ ] Reward config defines `critical_service_weights` summing to 1.0 for every scenario.
  - [ ] Reward config explicitly lists forbidden reward sources: evidence discovery, clue unlock, patch-id correctness, stage advancement, query success.
  - [ ] Existing scenario fixtures still load successfully.

  **QA Scenarios**:
  ```
  Scenario: Reward config loads for all scenarios
    Tool: Bash
    Steps: Run a Python one-liner importing all scenarios and validating weight sums and required keys.
    Expected: Exit 0; every scenario has complete reward config and valid normalized weights.
    Evidence: .sisyphus/evidence/task-1-reward-config.txt

  Scenario: Forbidden-source list is complete
    Tool: Bash
    Steps: Grep config and associated tests for all banned terms.
    Expected: Forbidden-source entries exist and are asserted in tests.
    Evidence: .sisyphus/evidence/task-1-reward-config-grep.txt
  ```

  **Commit**: YES | Message: `refactor(rewards): define shaping config and forbidden reward sources` | Files: `unified_incident_env/server/challenge.py`, nearby config module, tests

- [ ] 2. Extend `classify_vulnerability` into a structured hypothesis commit

  **What to do**: Modify `UnifiedIncidentAction` so `classify_vulnerability` carries a structured hypothesis payload: `vulnerability_type`, `affected_services`, `confidence`, and `recommended_next_action`. Update validators, observation/state mirrors if needed, and any schema-generation logic that relies on action fields.
  **Must NOT do**: Do not add `submit_hypothesis`; do not break existing parsing for valid old payloads without an explicit migration path.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: touches schema, parser expectations, and compatibility.
  - Skills: `[]`
  - Omitted: `[omarchy]`

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 7,8,9 | Blocked By: none

  **References**:
  - Pattern: `unified_incident_env/models.py:11-67` - current action schema.
  - Pattern: `unified_incident_env/trainer/prompts.py:216-230,385-405` - required-field and example generation.
  - Pattern: `unified_incident_env/tests/test_environment.py:333-345` - public action schema lock.
  - Pattern: `unified_incident_env/tests/test_trainer.py:45-107` - parser behavior expectations.

  **Acceptance Criteria**:
  - [ ] `classify_vulnerability` requires the new structured fields.
  - [ ] Existing explicit valid actions with complete fields parse successfully.
  - [ ] Tests cover missing `confidence`, malformed `affected_services`, and invalid recommended action values.

  **QA Scenarios**:
  ```
  Scenario: Structured hypothesis validates
    Tool: Bash
    Steps: Construct a valid classify_vulnerability action via Python and print model_dump.
    Expected: Exit 0; payload includes all structured hypothesis fields.
    Evidence: .sisyphus/evidence/task-2-hypothesis-valid.txt

  Scenario: Invalid hypothesis is rejected
    Tool: Bash
    Steps: Construct invalid actions missing required hypothesis fields.
    Expected: Validation raises deterministic errors.
    Evidence: .sisyphus/evidence/task-2-hypothesis-invalid.txt
  ```

  **Commit**: YES | Message: `feat(schema): structure vulnerability classification as scored hypothesis` | Files: `unified_incident_env/models.py`, parsers, tests

- [ ] 3. Implement incident-health potential helpers

  **What to do**: Add helper functions in `server/environment.py` (or a sibling reward helper module) to compute `operational_health`, `security_health`, and `incident_health_potential` from public/verified state only. Use service-status values `healthy=1.0`, `degraded=0.4`, `crashed=0.0`, weighted by scenario-authored critical-path weights.
  **Must NOT do**: Do not compute potential from evidence counters, stage names, recovery index, or hidden authored truth labels.

  **Recommended Agent Profile**:
  - Category: `quick` - Reason: local pure-function implementation once config is fixed.
  - Skills: `[]`
  - Omitted: `[omarchy]`

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 4 | Blocked By: 1

  **References**:
  - Pattern: `unified_incident_env/server/environment.py:501-516,556-560,692-700,856-947`
  - Pattern: `unified_incident_env/models.py:132-164,176-250`
  - External: Ng/Harada/Russell shaping paper above.

  **Acceptance Criteria**:
  - [ ] Potential helpers are pure and deterministic.
  - [ ] Potential increases when critical-path services improve.
  - [ ] Potential does not change from evidence-only discoveries when service/security health stays the same.

  **QA Scenarios**:
  ```
  Scenario: Potential rises on service recovery
    Tool: Bash
    Steps: Create before/after state fixtures with one critical service moving crashed -> healthy.
    Expected: after_potential > before_potential.
    Evidence: .sisyphus/evidence/task-3-potential-rise.txt

  Scenario: Evidence-only change has no positive shaping
    Tool: Bash
    Steps: Compare states that differ only by evidence counters/unlock flags.
    Expected: potential delta == 0.
    Evidence: .sisyphus/evidence/task-3-potential-no-breadcrumb.txt
  ```

  **Commit**: YES | Message: `refactor(rewards): add incident-health potential helpers` | Files: `unified_incident_env/server/environment.py`, tests

- [ ] 4. Rewrite environment step rewards around delta health + cost + penalties

  **What to do**: Replace per-handler positive breadcrumb rewards with a single post-transition reward computation based on `gamma * Φ(s') - Φ(s)`, normalized step cost, tiny hypothesis bonus/penalty, and explicit unsafe/redundant-action surcharges. Ensure repeated-action penalties flow through returned `reward`, not hidden cumulative mutations.
  **Must NOT do**: Do not keep direct `+0.05` query rewards, direct patch-id credit, or verify-button credit.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: central behavior change with many edge cases.
  - Skills: `[]`
  - Omitted: `[omarchy]`

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 10,11 | Blocked By: 1,3,6

  **References**:
  - Pattern: `unified_incident_env/server/environment.py:103-177,263-323,325-554,569-601`
  - Pattern: `unified_incident_env/tests/test_environment.py:205-232,307-330`
  - Pattern: `unified_incident_env/server/challenge.py` reward-relevant scenario metadata after Task 1.

  **Acceptance Criteria**:
  - [ ] Query/evidence actions emit only step cost or redundancy penalty unless the underlying world state improves.
  - [ ] Wrong/harmful actions emit negative reward.
  - [ ] Verified service recovery and exploit containment emit positive reward due to state improvement.
  - [ ] No hidden mutation adjusts cumulative reward independently of returned reward.

  **QA Scenarios**:
  ```
  Scenario: Investigation no longer gives breadcrumb reward
    Tool: Bash
    Steps: Run a fixed scenario reset then a single query action that only reveals evidence.
    Expected: reward <= 0, with no positive breadcrumb term.
    Evidence: .sisyphus/evidence/task-4-no-query-reward.txt

  Scenario: Verified recovery yields positive reward
    Tool: Bash
    Steps: Execute a known-good mitigation step that improves critical service health.
    Expected: reward > 0 and health potential increases.
    Evidence: .sisyphus/evidence/task-4-recovery-positive.txt
  ```

  **Commit**: YES | Message: `refactor(rewards): score steps by health delta and normalized costs` | Files: `unified_incident_env/server/environment.py`, tests

- [ ] 5. Rewrite public deterministic score to remove breadcrumb terms

  **What to do**: Update `server/grader.py` so `final_score` reflects verified operational recovery, verified security completion, efficiency, and postmortem quality without direct investigation-count or patch-id-guess terms. Preserve deterministic scoring/report shape.
  **Must NOT do**: Do not make public score depend on hidden health potential internals or trainer-specific gamma.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: public benchmark semantics change.
  - Skills: `[]`
  - Omitted: `[omarchy]`

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 11 | Blocked By: 1,6

  **References**:
  - Pattern: `unified_incident_env/server/grader.py:68-201`
  - Pattern: `unified_incident_env/tests/test_environment.py:349-388`

  **Acceptance Criteria**:
  - [ ] `relevant_investigations` is no longer part of `infrastructure_score`.
  - [ ] `selected_patch` or `selected_vulnerability` alone do not award public score before verification/completion.
  - [ ] Existing report/check structure remains deterministic.

  **QA Scenarios**:
  ```
  Scenario: Breadcrumb-only progress does not lift public score
    Tool: Bash
    Steps: Build a grader state with evidence collected but no verified containment/recovery.
    Expected: score remains low and below resolved benchmark thresholds.
    Evidence: .sisyphus/evidence/task-5-no-breadcrumb-public-score.txt

  Scenario: Verified containment and recovery dominate score
    Tool: Bash
    Steps: Compare partial state vs fully recovered/verified state in grader.
    Expected: fully recovered score > partial score.
    Evidence: .sisyphus/evidence/task-5-public-score-compare.txt
  ```

  **Commit**: YES | Message: `refactor(grader): remove breadcrumb terms from public score` | Files: `unified_incident_env/server/grader.py`, tests

- [ ] 6. Add scenario-authored reward metadata and critical-path weights

  **What to do**: Extend each scenario in `server/challenge.py` with deterministic critical-path service weights and reward metadata used by Tasks 3–5. Ensure these weights are scenario-local and normalized.
  **Must NOT do**: Do not infer weights dynamically from evidence or runtime guesses.

  **Recommended Agent Profile**:
  - Category: `quick`
  - Skills: `[]`
  - Omitted: `[omarchy]`

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 4,5 | Blocked By: 1

  **References**:
  - Pattern: `unified_incident_env/server/challenge.py:96-199,284-403,486-610`

  **Acceptance Criteria**:
  - [ ] Every scenario includes valid reward metadata.
  - [ ] Hard scenario weights emphasize worker/database path appropriately.
  - [ ] Tests verify normalization and required keys.

  **QA Scenarios**:
  ```
  Scenario: Scenario reward metadata validates
    Tool: Bash
    Steps: Import all scenarios and validate reward metadata shape.
    Expected: Exit 0; all scenarios satisfy schema.
    Evidence: .sisyphus/evidence/task-6-scenario-metadata.txt

  Scenario: Weight normalization is enforced
    Tool: Bash
    Steps: Sum critical_service_weights for each scenario.
    Expected: Each sum == 1.0 within tolerance.
    Evidence: .sisyphus/evidence/task-6-weight-sums.txt
  ```

  **Commit**: YES | Message: `feat(challenge): add critical-path service weights for reward shaping` | Files: `unified_incident_env/server/challenge.py`, tests

- [ ] 7. Update trainer prompt/schema generation for structured hypotheses

  **What to do**: Update `trainer/prompts.py` and parser-adjacent tests so `classify_vulnerability` examples and required fields include `affected_services`, `confidence`, and `recommended_next_action`. Fix the verification-stage mismatch explicitly if still present after schema changes.
  **Must NOT do**: Do not leak teacher actions into runtime prompts.

  **Recommended Agent Profile**:
  - Category: `quick`
  - Skills: `[]`
  - Omitted: `[omarchy]`

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: 8,9 | Blocked By: 2

  **References**:
  - Pattern: `unified_incident_env/trainer/prompts.py:96-148,385-434`
  - Pattern: `unified_incident_env/tests/test_trainer.py:229-253`

  **Acceptance Criteria**:
  - [ ] Runtime prompt examples for `classify_vulnerability` include the structured hypothesis payload.
  - [ ] `strict` and `lenient` behavior remain meaningfully distinct.
  - [ ] Verification-stage action table is internally consistent across environment and prompt schema.

  **QA Scenarios**:
  ```
  Scenario: Prompt shows structured hypothesis example
    Tool: Bash
    Steps: Build a runtime request in security_subquest stage.
    Expected: User prompt contains hypothesis fields and valid JSON example.
    Evidence: .sisyphus/evidence/task-7-prompt-hypothesis.txt

  Scenario: Strict mode remains stricter
    Tool: Bash
    Steps: Compare strict and lenient runtime requests with correction memory text.
    Expected: strict omits lenient correction hints.
    Evidence: .sisyphus/evidence/task-7-strict-vs-lenient.txt
  ```

  **Commit**: YES | Message: `feat(trainer): prompt structured vulnerability hypotheses` | Files: `unified_incident_env/trainer/prompts.py`, tests

- [ ] 8. Update inference fallback and schema handling for structured hypotheses

  **What to do**: Update `inference.py` so structured hypothesis payloads are generated, parsed, and repaired consistently. Keep the already-fixed verification-failure fallback behavior intact.
  **Must NOT do**: Do not reintroduce heuristic loops that bypass the new structured contract.

  **Recommended Agent Profile**:
  - Category: `unspecified-high`
  - Skills: `[]`
  - Omitted: `[omarchy]`

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: 12 | Blocked By: 2,7

  **References**:
  - Pattern: `inference.py:279-472,475-568,865-905,996-1094,1190-1241`
  - Pattern: `unified_incident_env/tests/test_submission_inference.py:99-166,205-355`

  **Acceptance Criteria**:
  - [ ] Fallback classification outputs valid structured hypotheses.
  - [ ] Repeated verification failures still return to patching.
  - [ ] Submission inference tests cover malformed hypothesis payloads.

  **QA Scenarios**:
  ```
  Scenario: Fallback builds structured hypothesis
    Tool: Bash
    Steps: Build fallback action in security_subquest before patching.
    Expected: classify_vulnerability action includes services, confidence, and next action fields.
    Evidence: .sisyphus/evidence/task-8-fallback-hypothesis.txt

  Scenario: Verification failure still re-patches
    Tool: Bash
    Steps: Reproduce failed verification state.
    Expected: narrowed actions and fallback choose apply_patch, not re-verify.
    Evidence: .sisyphus/evidence/task-8-repatch-after-failed-verify.txt
  ```

  **Commit**: YES | Message: `feat(inference): emit structured hypotheses and preserve safe fallback` | Files: `inference.py`, tests

- [ ] 9. Update baselines and walkthroughs for new hypothesis payload

  **What to do**: Update `scripts/baseline_agent.py`, walkthroughs, and any deterministic sample flows so they emit the structured `classify_vulnerability` action. Keep exact scenario solutions intact.
  **Must NOT do**: Do not alter scenario truth or recovery order here.

  **Recommended Agent Profile**:
  - Category: `quick`
  - Skills: `[]`
  - Omitted: `[omarchy]`

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: 10 | Blocked By: 2,7

  **References**:
  - Pattern: `unified_incident_env/scripts/baseline_agent.py`
  - Pattern: `unified_incident_env/scripts/walkthrough.py`
  - Pattern: `unified_incident_env/tests/test_environment.py` happy-path helpers.

  **Acceptance Criteria**:
  - [ ] Baseline agent still solves all three scenarios.
  - [ ] Structured hypothesis payload appears in the baseline classify step.

  **QA Scenarios**:
  ```
  Scenario: Baseline still solves preset pack
    Tool: Bash
    Steps: Run the baseline walkthrough or equivalent deterministic script/tests.
    Expected: All scenarios resolve successfully.
    Evidence: .sisyphus/evidence/task-9-baseline-solves.txt

  Scenario: Baseline classify step is structured
    Tool: Bash
    Steps: Print the classify_vulnerability payload from the baseline plan.
    Expected: Includes new hypothesis fields.
    Evidence: .sisyphus/evidence/task-9-baseline-structured-hypothesis.txt
  ```

  **Commit**: YES | Message: `refactor(baseline): emit structured classification hypotheses` | Files: baseline/walkthrough/tests

- [ ] 10. Add reward decomposition and anti-breadcrumb regression tests

  **What to do**: Add deterministic environment tests proving query/evidence actions no longer receive positive breadcrumb rewards and that repeated hypotheses do not farm reward.
  **Must NOT do**: Do not rely on broad “final score looks okay” assertions alone.

  **Recommended Agent Profile**:
  - Category: `unspecified-high`
  - Skills: `[]`
  - Omitted: `[omarchy]`

  **Parallelization**: Can Parallel: YES | Wave 4 | Blocks: Final verification | Blocked By: 4,9

  **References**:
  - Pattern: `unified_incident_env/tests/test_environment.py:205-232,235-372`

  **Acceptance Criteria**:
  - [ ] Pure evidence gathering has no positive breadcrumb reward.
  - [ ] Duplicate hypothesis submissions gain at most one bonus.
  - [ ] Harmful actions are negative.

  **QA Scenarios**:
  ```
  Scenario: Duplicate hypothesis bonus is one-time only
    Tool: Bash
    Steps: Submit same classify_vulnerability payload twice in a deterministic scenario.
    Expected: First bonus sign as designed; second bonus == 0 or negative cost only.
    Evidence: .sisyphus/evidence/task-10-hypothesis-dedupe.txt

  Scenario: Evidence-only step is non-positive
    Tool: Bash
    Steps: Reset then perform one diagnostic query.
    Expected: reward <= 0.
    Evidence: .sisyphus/evidence/task-10-evidence-nonpositive.txt
  ```

  **Commit**: YES | Message: `test(rewards): add anti-breadcrumb and hypothesis-dedupe regressions` | Files: environment tests

- [ ] 11. Add reward/public-score drift regression checks

  **What to do**: Create fixed-scenario comparisons proving that policies improving training reward also improve or at least align with public deterministic score ordering. Compare bad, partial, and good trajectories.
  **Must NOT do**: Do not require exact equality between training reward sums and final score.

  **Recommended Agent Profile**:
  - Category: `deep`
  - Skills: `[]`
  - Omitted: `[omarchy]`

  **Parallelization**: Can Parallel: YES | Wave 4 | Blocks: Final verification | Blocked By: 4,5

  **References**:
  - Pattern: `unified_incident_env/server/grader.py`
  - Pattern: `unified_incident_env/server/environment.py`
  - Pattern: existing happy/trap path tests in `tests/test_environment.py`.

  **Acceptance Criteria**:
  - [ ] Good trajectory > partial trajectory > harmful trajectory in public score.
  - [ ] Good trajectory accumulates better training reward than harmful trajectory.
  - [ ] No scenario shows breadcrumb-only trajectories outranking true containment/recovery.

  **QA Scenarios**:
  ```
  Scenario: Reward/public-score ordering aligns
    Tool: Bash
    Steps: Execute scripted bad, partial, and good trajectories for a fixed scenario.
    Expected: reward/public-score ordering is monotonic in the desired direction.
    Evidence: .sisyphus/evidence/task-11-ordering.txt

  Scenario: Breadcrumb trajectory cannot win
    Tool: Bash
    Steps: Run a query-heavy but unrecovered trajectory.
    Expected: Its public score and reward stay below a truly recovered trajectory.
    Evidence: .sisyphus/evidence/task-11-no-breadcrumb-win.txt
  ```

  **Commit**: YES | Message: `test(rewards): add reward-vs-public-score ordering checks` | Files: environment/grader tests

- [ ] 12. Document Colab/Kaggle GRPO usage with the new reward semantics

  **What to do**: Update docs/runbooks so training happens on Colab/Kaggle while the environment runs locally or via Docker. Explain the separation between training reward and public deterministic benchmark score, and point to the exact verification commands.
  **Must NOT do**: Do not leave the old reward explanation in README/execution docs.

  **Recommended Agent Profile**:
  - Category: `writing`
  - Skills: `[]`
  - Omitted: `[omarchy]`

  **Parallelization**: Can Parallel: YES | Wave 4 | Blocks: Final verification | Blocked By: 8

  **References**:
  - Pattern: `README.md`, `execution.md`, any training docs in repo.
  - External: `https://huggingface.co/docs/trl/en/openenv` - OpenEnv+TRL integration.

  **Acceptance Criteria**:
  - [ ] Docs explain training reward vs public score distinction.
  - [ ] Docs list the exact local test commands.
  - [ ] Docs specify Colab/Kaggle training and local/docker env execution.

  **QA Scenarios**:
  ```
  Scenario: Docs mention reward/public-score split
    Tool: Bash
    Steps: Grep updated docs for training reward, public score, and verification commands.
    Expected: All required topics present.
    Evidence: .sisyphus/evidence/task-12-doc-grep.txt

  Scenario: Docs commands are runnable
    Tool: Bash
    Steps: Execute at least one documented local verification command.
    Expected: Exit 0.
    Evidence: .sisyphus/evidence/task-12-doc-command.txt
  ```

  **Commit**: YES | Message: `docs(rewards): document shaping semantics and training workflow` | Files: docs/readme/runbooks

## Final Verification Wave (MANDATORY — after ALL implementation tasks)
> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback -> fix -> re-run -> present again -> wait for okay.
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Real Manual QA — unspecified-high (+ playwright if UI)
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy
- Commit 1: reward config + scenario metadata
- Commit 2: structured hypothesis schema
- Commit 3: health potential helpers
- Commit 4: environment reward rewrite
- Commit 5: grader rewrite
- Commit 6: prompt/inference/baseline contract updates
- Commit 7: regression tests + docs

## Success Criteria
- Training reward is driven by world-state improvement, not breadcrumb discovery.
- Public deterministic benchmark score no longer rewards evidence-count collection or raw patch-id guessing.
- `classify_vulnerability` supports calibrated, non-farmable hypothesis scoring.
- Query/evidence/unlock actions are not directly profitable.
- Verified containment + verified recovery dominate both reward and public score ordering.
- All tests and deterministic regression checks pass.
