# References — incidents, benchmarks, frameworks

The scenario design in sre-gym is grounded in a corpus of real 2022–2026 production incidents, surveyed against the gaps in existing AI-SRE benchmarks. This document is the receipts.

---

## 1. Incident corpus (where each Basic + Advanced + Max template comes from)

### Basic-tier templates and their grounding

| Template | Real-world incident pattern |
|---|---|
| `worker_deploy_cascade` | Classic "bad worker deploy → DB crash-loop → login 502s" pattern. Shape recurs across nearly every postmortem corpus. |
| `db_config_rollout` | Cloudflare Nov 2025 — bot-detection permissions regression caused by a config-rollout that shrunk database access. |
| `gateway_auth_rollout` | Base44 SaaS platform 2025 — auth-middleware deploy rejected ~40% of valid logins. |
| `payment_webhook_misconfig` | Stripe webhook signature drift class — API version mismatches breaking signature verification post-deploy. |
| `schema_drift_missing_migration` | Prisma/Supabase drift class — gateway code expects a column whose migration was never applied to prod. |
| `cache_stale_state` | TTL-bumping class — a "performance optimization" that increases TTL beyond session-lifetime, leading to cross-user state leaks. |
| `dep_degradation` | Cloudflare R2 Mar/Feb 2025 + Fly.io Apr 2026 disk-saturation class — dependency pool exhaustion. |
| `memory_leak_oom` | Worker OOM-restart-loop class — common across Node.js / Python deploys with long-lived buffers. |
| `auth_token_expiry` | Vercel Apr 2026 — third-party OAuth token expiry / Context.ai breach class. |
| `network_partition` | Cloudflare Aug/Sep 2025 + Fly.io Apr 2026 tunnel-hang class — DNS/discovery pin pointing at evicted target. |
| `rate_limit_retry_storm` | Stripe Mar 2022 retry-storm class — naïve retry policy amplifying upstream rate-limit. |
| `migration_lock` | Railway Oct 2025 — `CREATE INDEX` without `CONCURRENTLY` on billion-row table at peak. |

### Advanced-tier reference scenarios

| Scenario | Real-world pattern |
|---|---|
| `cascading_release_train` | Multi-stage: release-train deploys multiple services; one service's fix triggers a downstream incident in another. Common pattern in any monorepo with synchronized deploys. |
| `observability_pipeline_outage` | Cloudflare Nov 2025 — caught-exception storm saturating Loki + Promtail + the application's logging library. The pipeline becomes a denial-of-service vector. |
| `supabase_rls_silent_leak` | Supabase / Postgres RLS class — a refactor introduces a `USING (TRUE)` typo that silently leaks tenants. No SLO breach; only Sentry cardinality alerts and support tickets. |

### Max-tier chaos library

The 11 chaos patterns are listed and grounded individually in [`docs/MAX_TIER.md`](MAX_TIER.md) §3. Each pattern cites its real-world incident (Cloudflare Nov 2025, Railway Oct 2025, Vercel Apr 2026, Fly.io Oct 2024, etc.).

---

## 2. Existing AI-SRE benchmarks and their gaps

| Benchmark | Year | What it does | Gap sre-gym fills |
|---|---|---|---|
| [SRE-bench (Rootly)](https://github.com/Rootly-AI-Labs/SRE-skills-bench) | 2025 | MCQ-style declarative knowledge eval | **Not trainable.** Static eval, not RL env. |
| [agentkube/SRE-bench](https://github.com/agentkube/SRE-bench) | 2025 | SWE-bench-style, real K8s scenarios | **Requires K8s cluster.** Not runnable on cpu-basic HF Space. |
| [IBM ITBench](https://github.com/IBM/ITBench-SRE-Agent) | 2025 | 102 scenarios across SRE/FinOps/CISO | Framework-coupled (CrewAI). Static + live tiers but no compute-budget tiering. |
| [Microsoft AIOpsLab](https://github.com/microsoft/AIOpsLab) | 2024 | 48 problems on DeathStarBench microservices | Single-tier difficulty band; no explicit dimensional escalation. |
| [bugraid-ai/opensre-tools](https://github.com/bugraid-ai/opensre-tools) | 2024 | Generic infra failures | Doesn't specialize in vibe-coded SaaS specifically. |
| [microsoft/sre-agent](https://github.com/microsoft/sre-agent) | 2024 | Azure internal | Not open infrastructure. |
| [openenv-community/kube-sre-gym](https://huggingface.co/spaces/openenv-community/kube-sre-gym) | Apr 2026 | Kubernetes-cluster SRE | Doesn't cover the indie/SaaS layer (Stripe webhooks, Supabase RLS, schema drift). No tier story. |

The single thing sre-gym has that none of the above have is **the dimensional-escalation tier story** (compute → horizon → realism). The Basic + Advanced + Max design is structurally novel against the surveyed benchmarks.

---

## 3. Framework integration (OpenEnv)

OpenEnv features used by sre-gym Basic tier:

- `openenv.core.env_server.Environment[Action, Observation, State]` base class
- `openenv.core.Action / Observation / State` typed Pydantic models
- `openenv.core.env_server.http_server.create_fastapi_app` HTTP wiring
- `EnvironmentMetadata` for env discovery
- `max_concurrent_envs` for batched rollouts
- Custom `/tasks` `/baseline` `/grader` `/status` extension routes
- `openenv.core.EnvClient` for the test client

OpenEnv features documented for Advanced/Max but not implemented:

- `MCPEnvironment` base class (Advanced + Max would use this for production serving)
- `@self.tool()` action registration (Advanced + Max)
- `ServerMode.SIMULATION` vs `ServerMode.PRODUCTION` (Max specifically)
- WebSocket `/ws` transport (Advanced — for low-latency multi-agent rollouts)
- Custom Gradio `TabbedInterface` (Advanced + Max — topology inspector tabs)
- `openenv.cli` auto-discovery for the HF Spaces ecosystem (Basic does this; Advanced + Max would extend it)

---

## 4. Training framework integration (Unsloth + TRL + GRPO)

The Basic-tier training pipeline integrates:

- **Unsloth** for 4-bit / LoRA-friendly model loading (Qwen 2.5 3B)
- **HuggingFace TRL** for the GRPO loop, group-relative advantages, KL-control
- **OpenClaw-RL** pool-server pattern (`/allocate /reset /exec_tool /evaluate /close`) for async GRPO at training scale
- **Wandb** for training-curve logging

References:

- [Unsloth docs — Qwen2.5 LoRA](https://docs.unsloth.ai/get-started/all-our-models)
- [TRL — GRPO trainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [DeepMind GRPO paper](https://arxiv.org/abs/2402.03300)
- [Gen-Verse/OpenClaw-RL](https://github.com/Gen-Verse/OpenClaw-RL)

---

## 5. Vibe-coded SaaS security research (the framing for the SaaS-layer specialization)

The decision to specialize at the indie/vibe-coded SaaS layer (Stripe webhooks, Supabase RLS, schema drift, OAuth pivots) is grounded in:

- **Veracode 2025 AI Code Security Study** — 45% of AI-generated code has security flaws (n = 100+ LLMs, 80 scenarios)
- **JFrog / Snyk 2025** — ~40% of AI-generated database queries are SQL-injectable
- **Accorian 2025** — 88% of AI-generated logging unsafe; 86% of AI-generated input validation contains XSS errors
- **Replit / SaaStr incident, July 2025** — agent deleted production DB during an explicit code freeze
- **Tea app 2025** — leaked user data through unauthenticated admin routes
- **Base44 2025** — URI-construction bug let unauthenticated users hit privileged endpoints
- **Cloudflare Nov 2025** — bot-detection permissions regression (canonical config-rollout pattern)
- **Vercel Apr 2026** — Context.ai OAuth token compromise

The thesis is straightforward: vibe-coded SaaS is the fastest-shipping software category on Earth and has the weakest SRE muscle of any category ever shipped. An RL environment that specifically targets this failure surface is more valuable for training than a generic infra-failure simulator.

---

## 6. Persona-tiered benchmarking literature (the dimensional-escalation rationale)

The three-tier persona framing (Student / Startup / Enterprise) maps to the broader trend in 2023-2026 ML benchmarking literature:

- **SWE-bench Lite / Verified / Pro** — escalation along scenario count, single-file vs multi-file, languages
- **MLE-bench Low / Med / High / Lite** — escalation along dataset volume + horizon
- **ITBench static / live** — escalation along observability richness + execution risk
- **WebArena / -Verified / -Hard** — escalation along DOM volatility + horizon
- **CRMArena / -Pro** — escalation along multi-turn complexity + confidentiality

What's structurally new in sre-gym: each tier escalates a *different* dimension, not the same dimension at three depths. That's a strict generalization of the existing pattern, not a rebrand.

For the full survey, see the `Comprehensive Analysis of Compute-Budget-Tiered Machine Learning Benchmarks and Training Environments (2023–2026)` reference report compiled during the design phase.

---

## 7. Postmortem corpus consulted

The 45-incident postmortem corpus consulted during scenario design covers:

- **Cloudflare** — Feb/Jan 2026 config rollouts, Nov 2025 deploy regression, Oct/Jun 2023 routing/DNS, Aug/Sep 2025 dependency degradation, Mar 2025 R2 outage, Jun 2025 KV outage, Dec 2025 WAF, Feb 2025 R2 storage
- **Fly.io** — Apr 2026 SQLite mutex, Apr 2026 disk saturation, Apr 2026 cluster rebuild, Apr 2026 Sidekiq backlog, Apr 2026 fiber cut, Mar 2026 routing isolation, Dec 2024 proxy deadlock, Oct 2024 mesh storm, Apr 2026 tunnel hang
- **Railway** — Mar 2026 CDN contamination, Feb 2026 anti-fraud cascade, Feb 2026 DDoS, Jan 2026 GitHub rate-limit, Nov 2025 task queue, Oct 2025 Postgres lock, Dec 2025 framework CVE, Dec 2025 backend outage
- **Supabase** — Feb 2026 VPC blackout, Feb 2026 edge function 504s
- **Netlify** — Mar/Apr 2026 deploy + dependency regressions, Feb 2026 query contention
- **Stripe** — Mar 2022 latency / retry-storm, Feb 2024 ledger drop, Sep 2025 streaming patches
- **Vercel** — Apr 2026 OAuth pivot, Oct 2025 metadata routing, Mar/Apr 2026 Edge function regressions
- **PlanetScale** — Oct 2025 us-east-1 cascade

For each incident: time-to-detect, time-to-mitigate, time-to-resolve, diagnostic signals, investigative red herrings, eventual remediation, and reversibility.

The full corpus was used to derive the 12 Basic templates' fault patterns, the Advanced reference scenarios' multi-incident chains, and the Max chaos library's 11 patterns. See the corresponding tier docs for the per-template / per-scenario / per-pattern grounding citations.

---

## 8. Suggested reading order for skeptical reviewers

If you have 5 minutes:

1. README.md first paragraph (the dimensional-escalation insight)
2. README.md Basic-tier template table (the 12 templates and what each teaches)
3. `notebooks/02_basic_eval_comparison.ipynb` last cell (the comparison table)

If you have 15 minutes, add:

4. `docs/ARCHITECTURE.md` §1 + §3 (full dimensional escalation defence + Basic deep dive)
5. `docs/REWARD_DESIGN.md` §3-§4 (shaped rewards + hardened ceiling rationale)
6. One Advanced YAML (`cascading_release_train.yaml` is the most novel)

If you have 45 minutes, add:

7. `docs/BASIC_TIER.md` (full template + simulator + procgen detail)
8. `docs/ADVANCED_TIER.md` + `docs/MAX_TIER.md` (the design space)
9. `unified_incident_env/server/grader.py` (the actual rubric implementation)
10. `unified_incident_env/server/environment.py` (the world simulator)
