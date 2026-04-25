"""``python -m sre_gym.strategy`` entry-point. Delegates to runner.main()."""

from __future__ import annotations

from sre_gym.strategy.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
