"""
app.py — OpenEnv-compliant FastAPI server wrapping IncidentResponseEnv.

Endpoints
---------
  POST /reset        body: {"task_id": str, "seed": int|null}
  POST /step         body: Action JSON
  GET  /state        returns full internal snapshot

One environment instance lives per process. For concurrent evaluation the
OpenEnv validator spawns separate containers, so single-instance state is fine.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


from env.models import Action, Observation
from env.tasks import list_tasks
from env.env import IncidentResponseEnv
# ---------------------------------------------------------------------------
# App + shared env instance
# ---------------------------------------------------------------------------

app = FastAPI(title="IncidentResponseEnv", version="1.0.0")

_env: IncidentResponseEnv | None = None


def _get_env() -> IncidentResponseEnv:
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialised — call POST /reset first.")
    return _env


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "auth-crash-loop"
    seed:    int | None = 42


# ---------------------------------------------------------------------------
# Response bodies
# ---------------------------------------------------------------------------

class ResetResponse(BaseModel):
    observation: dict[str, Any]


class StepResponse(BaseModel):
    observation: dict[str, Any]
    reward:      float
    done:        bool
    info:        dict[str, Any]


class StateResponse(BaseModel):
    state: dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest) -> ResetResponse:
    global _env
    _env = IncidentResponseEnv(task_id=req.task_id, seed=req.seed)
    obs: Observation = _env.reset()
    return ResetResponse(observation=obs.model_dump())


@app.post("/step", response_model=StepResponse)
def step(action: Action) -> StepResponse:
    env = _get_env()
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return StepResponse(
        observation=obs.model_dump(),
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    env = _get_env()
    return StateResponse(state=env.state())


# ---------------------------------------------------------------------------
# Health + task listing (convenience, not required by OpenEnv spec)
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks")
def tasks() -> dict[str, Any]:
    return {"tasks": list_tasks()}