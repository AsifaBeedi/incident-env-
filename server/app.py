from __future__ import annotations
from typing import Any

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

from env.models import Action, Observation
from env.tasks import list_tasks
from env.env import IncidentResponseEnv

app = FastAPI(title="IncidentResponseEnv", version="1.0.0")

_env: IncidentResponseEnv | None = None


def _get_env() -> IncidentResponseEnv:
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return _env


class ResetRequest(BaseModel):
    task_id: str = "auth-crash-loop"
    seed: int | None = 42


@app.post("/reset")
def reset(req: ResetRequest = Body(default=ResetRequest())):
    global _env
    _env = IncidentResponseEnv(task_id=req.task_id, seed=req.seed)
    obs: Observation = _env.reset()
    return {"observation": obs.model_dump()}


@app.post("/step")
def step(action: Action):
    env = _get_env()
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    env = _get_env()
    return {"state": env.state()}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def tasks():
    return {"tasks": list_tasks()}


# 🔥 REQUIRED FOR VALIDATION
def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()