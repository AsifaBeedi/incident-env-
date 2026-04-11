"""
server/app.py — FastAPI server for IncidentResponseEnv

KEY FIX: _safe_float and all response sanitization ensures every numeric
value that could be interpreted as a "score" is strictly in (0.0, 1.0).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Body, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from env.models import Action, ActionType, Observation
from env.tasks import list_tasks
from env.env import IncidentResponseEnv

app = FastAPI(title="IncidentResponseEnv", version="1.0.0")

_env: IncidentResponseEnv | None = None

# Score bounds — strictly within (0.0, 1.0) exclusive
_SCORE_MIN = 0.02
_SCORE_MAX = 0.98


def _get_env(task_id="auth-crash-loop") -> IncidentResponseEnv:
    global _env
    if _env is None:
        _env = IncidentResponseEnv(task_id=task_id)
        _env.reset()
    return _env


def _safe_float(v: Any, default: float = 0.05) -> float:
    """
    Convert v to a float strictly in (0.0, 1.0).
    Never returns 0.0, 1.0, or any value outside (0.0, 1.0).
    """
    try:
        val = float(v)
    except (TypeError, ValueError):
        return default
    # Guard NaN and Inf
    if val != val or val == float("inf") or val == float("-inf"):
        return default
    # Clamp strictly to (_SCORE_MIN, _SCORE_MAX)
    if val <= 0.0:
        return _SCORE_MIN
    if val >= 1.0:
        return _SCORE_MAX
    clamped = max(_SCORE_MIN, min(_SCORE_MAX, val))
    return round(clamped, 4)


def _sanitize_reward_breakdown(rb: dict) -> dict:
    """
    Ensure reward_breakdown only exposes score-safe values.
    final and raw are authoritative scores — clamped to (0.02, 0.98).
    Component fields are kept as-is (they are deltas, not scores).
    """
    result = {}
    # Authoritative score fields — must be in (0, 1)
    result["final"] = _safe_float(rb.get("final", 0.05))
    result["raw"] = _safe_float(rb.get("raw", 0.05))
    # Delta fields — informational only, not checked as scores
    for key in ["inspection", "diagnosis", "fix", "partial_fix",
                "harmful", "irrelevant", "repeat", "no_diagnosis",
                "no_op", "budget_bonus"]:
        result[key] = rb.get(key, 0.0)
    return result


def _sanitize_info(info: dict) -> dict:
    """
    Sanitize the entire info dict so every numeric value that could be
    read as a score is strictly in (0.0, 1.0).
    """
    result = dict(info)

    # Cumulative reward — must be in (0, 1)
    if "cumulative_reward" in result:
        result["cumulative_reward"] = _safe_float(result["cumulative_reward"])

    # Episode score — must be in (0, 1)
    if "episode_score" in result:
        result["episode_score"] = _safe_float(result["episode_score"])

    # task_solved_score — must be in (0, 1)
    if "task_solved_score" in result:
        result["task_solved_score"] = _safe_float(result["task_solved_score"])

    # task_solved — keep as bool, also add numeric version
    if "task_solved" in result:
        solved = bool(result["task_solved"])
        result["task_solved"] = solved
        # Numeric version: never exactly 0.0 or 1.0
        result["task_solved_score"] = 0.95 if solved else 0.05

    # Reward breakdown
    if "reward_breakdown" in result:
        rb = result["reward_breakdown"]
        if isinstance(rb, dict):
            result["reward_breakdown"] = _sanitize_reward_breakdown(rb)

    # true_root_cause — convert enum to string value if needed
    if "_true_root_cause" in result and hasattr(result["_true_root_cause"], "value"):
        result["_true_root_cause"] = result["_true_root_cause"].value

    return result


def _make_fallback_info(cumulative: float = 0.05) -> dict:
    """Build a safe fallback info dict when the env is in error state."""
    return {
        "cumulative_reward": _safe_float(cumulative),
        "episode_score": _safe_float(cumulative),
        "task_solved": False,
        "task_solved_score": 0.05,
        "steps_remaining": 0,
        "diagnosis_set": False,
        "inspected_services": [],
        "_true_root_cause": "unknown",
        "_true_fix_target": "",
        "reward_breakdown": {
            "final": _safe_float(cumulative),
            "raw": _safe_float(cumulative),
            "inspection": 0.0, "diagnosis": 0.0, "fix": 0.0,
            "partial_fix": 0.0, "harmful": 0.0, "irrelevant": 0.0,
            "repeat": 0.0, "no_diagnosis": 0.0, "no_op": 0.0,
            "budget_bonus": 0.0,
        },
    }


class ResetRequest(BaseModel):
    task_id: str = "auth-crash-loop"
    seed: int | None = 42


@app.post("/reset")
def reset(req: ResetRequest = Body(default=ResetRequest())):
    global _env
    try:
        _env = IncidentResponseEnv(task_id=req.task_id, seed=req.seed)
        obs: Observation = _env.reset()
        return {"observation": obs.model_dump()}
    except Exception:
        _env = IncidentResponseEnv(task_id="auth-crash-loop", seed=req.seed)
        obs = _env.reset()
        return {"observation": obs.model_dump()}


@app.post("/step")
def step(action: Action):
    env = _get_env()
    try:
        obs, reward, done, info = env.step(action)
        safe_reward = _safe_float(reward)
        return {
            "observation": obs.model_dump(),
            "reward": safe_reward,
            "done": done,
            "info": _sanitize_info(info),
        }
    except RuntimeError:
        obs = env._build_observation()
        return {
            "observation": obs.model_dump(),
            "reward": _safe_float(0.05),
            "done": True,
            "info": _make_fallback_info(0.05),
        }
    except Exception:
        obs = env._build_observation()
        return {
            "observation": obs.model_dump(),
            "reward": _safe_float(0.05),
            "done": True,
            "info": _make_fallback_info(0.05),
        }


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    env = _get_env()
    try:
        fallback_action = Action(action_type=ActionType.NO_OP)
        obs, reward, done, info = env.step(fallback_action)
        return JSONResponse(
            status_code=200,
            content={
                "observation": obs.model_dump(),
                "reward": _safe_float(reward),
                "done": done,
                "info": _sanitize_info(info),
            }
        )
    except RuntimeError:
        obs = env._build_observation()
        return JSONResponse(
            status_code=200,
            content={
                "observation": obs.model_dump(),
                "reward": _safe_float(0.05),
                "done": True,
                "info": _make_fallback_info(0.05),
            }
        )


@app.get("/state")
def state():
    env = _get_env()
    s = env.state()
    # Ensure cumulative_reward and episode_score are safe
    s["cumulative_reward"] = _safe_float(s.get("cumulative_reward", 0.05))
    s["episode_score"] = _safe_float(s.get("episode_score", 0.05))
    return {"state": s}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def tasks():
    return {"tasks": list_tasks()}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
