"""python -m reward_rl.evaluate -- runs evaluation CLI (Section 25)."""

import sys


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")


_configure_utf8_stdio()

# Import directly from the module to avoid circular import through __init__
from reward_rl.evaluation.evaluate import main

if __name__ == "__main__":
    main()
