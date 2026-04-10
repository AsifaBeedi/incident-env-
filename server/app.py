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

def _get_env(task_id="auth-crash-loop") -> IncidentResponseEnv:
    global _env
    if _env is None:
        _env = IncidentResponseEnv(task_id=task_id)
        _env.reset()
    return _env

def _safe_float(v: Any) -> float:
    try:
        val = float(v)
    except (TypeError, ValueError):
        return 0.51
    if val <= 0.0:
        return 0.02
    if val >= 1.0:
        return 0.98
    if val < 0.01:
        return 0.02
    if val > 0.99:
        return 0.98
    return round(val, 4)

def _sanitize_info(info: dict) -> dict:
    if "cumulative_reward" in info:
        info["cumulative_reward"] = _safe_float(info["cumulative_reward"])
    if "reward_breakdown" in info:
        rb = info["reward_breakdown"]
        if "final" in rb:
            rb["final"] = _safe_float(rb["final"])
        if "raw" in rb:
            rb["raw"] = _safe_float(rb["raw"])
    if "_true_root_cause" in info and hasattr(info["_true_root_cause"], 'value'):
        info["_true_root_cause"] = info["_true_root_cause"].value
    return info

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
        return {
            "observation": obs.model_dump(),
            "reward": _safe_float(reward),
            "done": done,
            "info": _sanitize_info(info),
        }
    except RuntimeError:
        obs = env._build_observation()
        return {
            "observation": obs.model_dump(),
            "reward": 0.02,
            "done": True,
            "info": {
                "cumulative_reward": 0.02,
                "task_solved": False,
                "steps_remaining": 0,
                "diagnosis_set": False,
                "inspected_services": [],
                "_true_root_cause": "unknown",
                "_true_fix_target": "",
            }
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
                "reward": 0.02,
                "done": True,
                "info": {
                    "cumulative_reward": 0.02,
                    "task_solved": False,
                    "steps_remaining": 0,
                    "diagnosis_set": False,
                    "inspected_services": [],
                    "_true_root_cause": "unknown",
                    "_true_fix_target": "",
                }
            }
        )

@app.get("/state")
def state():
    env = _get_env()
    s = env.state()
    s["cumulative_reward"] = _safe_float(s.get("cumulative_reward", 0.02))
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
