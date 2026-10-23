"""Metrics, ground-truth success evaluators, and benchmark runners (Section 24: evaluation/).

Imports are lazy to avoid circular imports with envs/ (key_door_env imports
success_evaluator, and envs/__init__ triggers this __init__).
"""


def __getattr__(name):
    if name in ("EvaluationResult", "FailureReason", "evaluate_episode"):
        from reward_rl.evaluation.success_evaluator import (
            EvaluationResult,
            FailureReason,
            evaluate_episode,
        )
        _exports = {
            "EvaluationResult": EvaluationResult,
            "FailureReason": FailureReason,
            "evaluate_episode": evaluate_episode,
        }
        globals().update(_exports)
        return _exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["EvaluationResult", "FailureReason", "evaluate_episode"]
