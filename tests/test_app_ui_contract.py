from __future__ import annotations

import importlib


ui_app = importlib.import_module("app")


def test_requested_qwen_defaults_are_exposed() -> None:
    assert ui_app.TIER_DEFAULT_MODEL == {
        "basic": "Qwen/Qwen2.5-7B-Instruct",
        "advanced": "Qwen/Qwen2.5-72B-Instruct",
        "max": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    }


def test_run_enabled_requires_both_token_and_model() -> None:
    assert not ui_app._run_enabled("", "")
    assert not ui_app._run_enabled("hf_test", "")
    assert not ui_app._run_enabled("", "Qwen/Qwen2.5-7B-Instruct")
    assert ui_app._run_enabled("hf_test", "Qwen/Qwen2.5-7B-Instruct")


def test_suggest_model_uses_requested_defaults_for_blank_or_previous_default() -> None:
    assert ui_app._suggest_model("basic", "") == "Qwen/Qwen2.5-7B-Instruct"
    assert ui_app._suggest_model("advanced", "Qwen/Qwen2.5-7B-Instruct") == "Qwen/Qwen2.5-72B-Instruct"
    assert ui_app._suggest_model("max", "Qwen/Qwen2.5-72B-Instruct") == "Qwen/Qwen3-235B-A22B-Instruct-2507"


def test_suggest_model_preserves_manual_override() -> None:
    assert ui_app._suggest_model("advanced", "meta-llama/Llama-3.1-8B-Instruct") == "meta-llama/Llama-3.1-8B-Instruct"


def test_basic_target_resolution_uses_category_catalog() -> None:
    target, error = ui_app._resolve_target(ui_app.Tier.BASIC, "deploy", "")
    assert error is None
    assert target == "worker_deploy_cascade"
