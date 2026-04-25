"""``python -m sre_gym.operations`` entry-point. Delegates to runner.main()."""

from __future__ import annotations

from sre_gym.operations.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
