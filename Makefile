.PHONY: install dev test baseline walkthrough trainer-eval trainer-dataset trainer-session docker-build docker-run validate clean

install:
	python3 -m pip install -e ".[dev]"
	@echo "Dependencies installed"

dev:
	ENABLE_WEB_INTERFACE=true uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

test:
	pytest unified_incident_env/tests -v --tb=short

baseline:
	python -m unified_incident_env.scripts.baseline_agent

walkthrough:
	python -m unified_incident_env.scripts.walkthrough --scenario easy_sqli_db_outage

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
