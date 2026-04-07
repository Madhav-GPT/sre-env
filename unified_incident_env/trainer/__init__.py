"""Training scaffold for unified incident environment."""

from .action_adapter import LenientActionAdapter, StrictActionParser
from .analyze_failures import analyze_block, analyze_episode
from .backend import OpenAICompatibleBackend
from .collect_trajectory import collect_episode
from .policy_adapter import PolicyAdapter
from .run_session import run_session
from .run_episode import EpisodeRunner, run_episode
from .train_external import build_policy_adapter

__all__ = [
    "EpisodeRunner",
    "LenientActionAdapter",
    "OpenAICompatibleBackend",
    "PolicyAdapter",
    "StrictActionParser",
    "analyze_block",
    "analyze_episode",
    "build_policy_adapter",
    "collect_episode",
    "run_episode",
    "run_session",
]
