# SRE Engineer LLM (v2): The Honest SRE Simulator

`sre-engineer-llm` is a high-fidelity Reinforcement Learning (RL) environment designed to train and evaluate AI agents on **Site Reliability Engineering (SRE)** and **Incident Response**.

Unlike traditional "scripted" environments, this benchmark uses an honest, world-state-based simulation where agents must diagnose, mitigate, and resolve production outages without "cheating" through prompt oracles or hardcoded rails.

## 🚀 Key Features

- **Honest Simulation:** No stage-locks or hidden oracles. All actions are available at all times.
- **State-Based Transitions:** Remediation actions (like `rollback_deploy` or `restart_service`) directly affect the health metrics of the simulated services.
- **Verification Driven:** Agents must explicitly run health checks (`run_check`) to verify recovery before declaring an incident resolved.
- **Realistic SRE Stack:** Includes queries for logs, metrics, dependencies, and deployment history across a microservices topology.
- **Deterministic Grading:** A transparent scoring system based on final system health, user impact, and operational efficiency.

## 🛠 Action Space

The agent has access to 11 discrete SRE tools:

| Action | Description |
| :--- | :--- |
| `query_logs` | Inspect service-level error logs and traces. |
| `query_metrics` | Retrieve CPU, Memory, or Latency data. |
| `query_dependencies` | Map upstream and downstream service links. |
| `query_deploys` | Check the deployment history for recent changes. |
| `rollback_deploy` | Revert a service to its previous stable version. |
| `restart_service` | Reboot a crashed or degraded service. |
| `isolate_service` | Cut traffic to a service to contain blast radius. |
| `submit_hypothesis` | Record a calibrated guess of the root cause. |
| `run_check` | Execute a health/verification check on the system. |
| `declare_resolved` | Finalize the incident after recovery is verified. |
| `escalate` | Request expert attention (no-op in simulation). |

## 📁 Project Structure

- `unified_incident_env/`
  - `server/`: The FastAPI-based environment server.
    - `environment.py`: Core simulator logic and world-state transitions.
    - `challenge.py`: Scenario catalog and baseline definitions.
    - `grader.py`: Deterministic scoring and reporting logic.
  - `models.py`: Pydantic schemas for Actions, Observations, and State.
  - `client.py`: Typed client for interacting with the environment.
- `inference.py`: Standard entrypoint for LLM-based agent evaluation.
- `run_demo.py`: End-to-end script to run the server and the baseline agent.

## 🚦 Quick Start

### 1. Install Dependencies
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### 2. Run the Benchmark Demo
This script launches the local server and executes the optimal "baseline" trajectory:
```bash
python run_demo.py
```

### 3. Run Tests
```bash
pytest unified_incident_env/tests -q
```

## 📊 Scoring Breakdown

Success is measured across four primary dimensions:
1.  **Recovery (45%):** Is the end-to-end system healthy and the cause removed?
2.  **Security/Mitigation (35%):** Was the correct remediation target identified and fixed?
3.  **Efficiency (10%):** Did the agent solve the incident within the tick budget without wasteful actions?
4.  **Verification (10%):** Were all health checks passed before resolution?

## 📝 License
This project is licensed under the MIT License.
