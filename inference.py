from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI
from env import IncidentResponseEnv, Action, ActionType, DiagnosisTag, list_tasks

# ---------------- CONFIG ----------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1").strip()
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct").strip()
HF_TOKEN     = os.getenv("HF_TOKEN", "").strip()

ENV_NAME = "IncidentResponseEnv"
SEED = 42

_CLIENT = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ---------------- VALID VALUES ----------------
_VALID_ACTION_TYPES = [a.value for a in ActionType]
_VALID_DIAGNOSES    = [d.value for d in DiagnosisTag]

# ---------------- PROMPT ----------------
_SYSTEM_PROMPT = f"""
You are an expert SRE handling incidents.

Available action_type:
{json.dumps(_VALID_ACTION_TYPES)}

Available diagnosis:
{json.dumps(_VALID_DIAGNOSES)}

Rules:
- ALWAYS include both action_type and target
- Inspect → Diagnose → Fix
- Never repeat actions
- Never fix without diagnosis

Return ONLY JSON.
"""

# ---------------- OBS ----------------
def _obs_to_text(obs: Any) -> str:
    payload = {
        "step": obs.step,
        "services": [
            {
                "name": s.name,
                "status": s.status.value,
                "cpu": s.metrics.cpu_usage,
                "mem": s.metrics.memory_usage,
                "error": s.metrics.error_rate,
            }
            for s in obs.services
        ],
        "logs": [f"{e.service}: {e.message}" for e in obs.logs],
    }
    return json.dumps(payload)

# ---------------- MODEL ----------------
def _call_model(messages):
    try:
        res = _CLIENT.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=200,
        )
        raw = res.choices[0].message.content.strip()
    except Exception as e:
        return Action(action_type=ActionType.INSPECT_LOGS, target="auth-service"), str(e)

    try:
        data = json.loads(raw)
    except:
        return Action(action_type=ActionType.INSPECT_LOGS, target="auth-service"), "json_error"

    atype = data.get("action_type")
    target = data.get("target")

    # 🔥 HARD GUARDRAILS
    if atype not in _VALID_ACTION_TYPES:
        return Action(action_type=ActionType.INSPECT_LOGS, target="auth-service"), "invalid_action"

    if not target or not isinstance(target, str):
        target = "auth-service"

    params = data.get("parameters", {}) or {}

    return Action(
        action_type=ActionType(atype),
        target=target,
        parameters=params,
    ), None

# ---------------- EPISODE ----------------
def _run_episode(task_id) -> bool:
    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    env = IncidentResponseEnv(task_id=task_id, seed=SEED)
    obs = env.reset()

    rewards = []
    step = 0
    solved = False
    api_failed = False

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    max_steps = 6

    for _ in range(max_steps):
        messages.append({"role": "user", "content": _obs_to_text(obs)})

        action, error = _call_model(messages)

        obs, reward, done, info = env.step(action)

        step += 1
        rewards.append(reward)

        if done and info.get("task_solved"):
            solved = True

        action_str = action.action_type.value
        if action.target:
            action_str += f":{action.target}"

        error_str = error.replace("\n", " ") if error else "null"

        print(
            f"[STEP] step={step} action={action_str} "
            f"reward={reward:.2f} done={'true' if done else 'false'} error={error_str}",
            flush=True,
        )

        # STOP on API failure
        if error and ("402" in error or "rate" in error.lower() or "timeout" in error.lower()):
            api_failed = True
            break

        if done:
            break

        messages.append({"role": "assistant", "content": json.dumps({
            "action_type": action.action_type.value,
            "target": action.target
        })})

    print(
        f"[END] success={'true' if solved else 'false'} steps={step} "
        f"rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )

    return api_failed

# ---------------- MAIN ----------------
def main():
    for task in list_tasks():
        failed = _run_episode(task["id"])
        if failed:
            break

if __name__ == "__main__":
    main()