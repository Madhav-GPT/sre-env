"""OpenClaw-RL integration shim for sre-gym.

This package exposes sre-gym through the lease-based HTTP contract used by
OpenClaw-RL's `terminal-rl/` and `swe-rl/` training loops, so the existing
OpenClaw-RL rollout+training scripts can target this env without code forks.
"""
