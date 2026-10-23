"""python -m reward_rl.train -- runs training CLI (Section 25)."""

import sys


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")


_configure_utf8_stdio()

from reward_rl.training.train import main

if __name__ == "__main__":
    main()
