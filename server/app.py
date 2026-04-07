"""Top-level OpenEnv entrypoint wrapper."""

from unified_incident_env.server.app import app, serve
from unified_incident_env.server.app import main as _main

__all__ = ["app", "main", "serve"]


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
