.PHONY: install dev test test-tier1 test-wrapper baseline scripted-baseline tier-info walkthrough trainer-eval trainer-dataset trainer-session docker-build docker-run validate clean

install:
	python3 -m pip install -e ".[dev]"
	@echo "Dependencies installed (sre-engineer-llm 3.1.0 — Basic tier runnable, Advanced/Max runnable as Python orchestrators with design-spec YAMLs)"

install-train:
	python3 -m pip install -e ".[dev,train]"
	@echo "Train deps installed (Unsloth + TRL pulled separately in notebook 01)"

dev:
	ENABLE_WEB_INTERFACE=true uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

test:
	pytest unified_incident_env/tests tests/ -v --tb=short

test-tier1:
	pytest unified_incident_env/tests/test_round2_templates.py -v --tb=short

test-wrapper:
	pytest tests/test_sre_gym_wrapper.py -v --tb=short

baseline:
	python scripts/eval_baseline.py --templates-only

scripted-baseline:
	python scripts/eval_baseline.py --output eval/results/scripted_baseline.jsonl

tier-info:
	python -c "from sre_gym import SREGym, Tier; \
		[print(f'\\n=== {t.value} ==='), [print(f'  {k:<28} {v}') for k,v in SREGym(tier=t).describe().items()]] for t in Tier"

walkthrough:
	python -m unified_incident_env.scripts.walkthrough --scenario worker_deploy_cascade

trainer-eval:
	python -m unified_incident_env.trainer.eval_models --models qwen2.5:0.5b gemma2:2b qwen2.5:7b-instruct-q4_K_M --mode strict

trainer-dataset:
	python -m unified_incident_env.trainer.build_sft_dataset --source combined --output outputs/trainer/sft_dataset.jsonl

trainer-session:
	python -m unified_incident_env.trainer.run_session --model qwen2.5:0.5b --base-url http://127.0.0.1:8000

docker-build:
	docker buildx build --platform linux/amd64 -t sre-env:latest .

docker-run:
	docker run -p 8000:8000 -e ENABLE_WEB_INTERFACE=true sre-env:latest

validate:
	openenv validate .

clean:
	rm -rf outputs __pycache__ .pytest_cache
	find . -name "*.pyc" -delete
