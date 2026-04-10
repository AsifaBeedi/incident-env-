from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from env.env    import IncidentResponseEnv
from env.models import Action, ActionType, DiagnosisTag
from env.tasks  import list_tasks

# ---------------- CONFIG ----------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1").strip()
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-7B-Instruct").strip()
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

ENV_NAME = "IncidentResponseEnv"
SEED     = 42

_CLIENT = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ---------------- VALID VALUES ----------------
_VALID_ACTION_TYPES = [a.value for a in ActionType]
_VALID_DIAGNOSES    = [d.value for d in DiagnosisTag]

# ---------------- PROMPT ----------------
_SYSTEM_PROMPT = f"""You are an expert SRE handling an incident.
Available action_type: {json.dumps(_VALID_ACTION_TYPES)}
Available diagnosis: {json.dumps(_VALID_DIAGNOSES)}
Rules:
- ALWAYS include both action_type and target
- Inspect first, then diagnose, then fix
- Never repeat the same (action_type, target) pair
- Never fix without a prior diagnosis
Return ONLY a JSON object like:
{{"action_type": "inspect_logs", "target": "auth-service", "parameters": {{}}}}"""


# ---------------- OBS ----------------
def _obs_to_text(obs: Any) -> str:
    payload = {
        "step":    obs.step,
        "alerts":  obs.active_alerts,
        "services": [
            {
                "name":    s.name,
                "status":  s.status.value,
                "cpu":     round(s.metrics.cpu_usage, 1),
                "mem":     round(s.metrics.memory_usage, 1),
                "errors":  round(s.metrics.error_rate, 1),
                "latency": round(s.metrics.latency_ms, 1),
            }
            for s in obs.services
        ],
        "logs":          [f"[{e.severity.value.upper()}] {e.service}: {e.message}"
                          for e in obs.logs[:10]],
        "inspected":     obs.inspected,
        "diagnosis_set": obs.diagnosis_set,
    }
    return json.dumps(payload)


# ---------------- MODEL CALL ----------------
def _call_model(messages: list[dict]) -> tuple[Action, str | None]:
    try:
        res = _CLIENT.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=200,
        )
        raw = res.choices[0].message.content.strip()
    except Exception as exc:
        return Action(action_type=ActionType.NO_OP, target=None), str(exc)

    if raw.startswith("```"):
        raw = "\n".join(l for l in raw.splitlines() if not l.startswith("```")).strip()

    try:
        data = json.loads(raw)
    except Exception:
        return Action(action_type=ActionType.NO_OP, target=None), "json_parse_error"

    atype  = data.get("action_type", "")
    target = data.get("target") or None
    params = data.get("parameters", {}) or {}

    if atype not in _VALID_ACTION_TYPES:
        return Action(action_type=ActionType.NO_OP, target=None), f"invalid_action:{atype!r}"

    diag = params.get("diagnosis")
    if diag is not None:
        if diag not in _VALID_DIAGNOSES:
            params.pop("diagnosis")
        else:
            params["diagnosis"] = DiagnosisTag(diag)

    return Action(action_type=ActionType(atype), target=target, parameters=params), None


# ---------------- EPISODE ----------------
def _run_episode(task_id: str) -> None:
    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    rewards:      list[float] = []
    step:         int         = 0
    solved:       bool        = False
    seen_actions: set         = set()
    env = None

    try:
        task_meta = next(t for t in list_tasks() if t["id"] == task_id)
        max_steps = task_meta["max_steps"]

        env = IncidentResponseEnv(task_id=task_id, seed=SEED)
        obs = env.reset()
        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

        for _ in range(max_steps):
            messages.append({"role": "user", "content": _obs_to_text(obs)})
            action, error = _call_model(messages)

            # Deduplicate actions
            action_key = (action.action_type.value, action.target)
            if action_key in seen_actions:
                action = Action(action_type=ActionType.NO_OP, target=action.target)
            else:
                seen_actions.add(action_key)

            obs, reward, done, info = env.step(action)
            step += 1
            rewards.append(reward)

            if done and info.get("task_solved") is True:
                solved = True

            # Build action string
            action_str = action.action_type.value
            if action.target:
                action_str += f":{action.target}"
            diag = action.parameters.get("diagnosis")
            if diag:
                diag_val = diag.value if hasattr(diag, "value") else str(diag)
                action_str += f"[{diag_val}]"

            error_str = error.replace("\n", " ").strip() if error else "null"

            print(
                f"[STEP] step={step} action={action_str} "
                f"reward={reward:.2f} done={'true' if done else 'false'} "
                f"error={error_str}",
                flush=True,
            )

            messages.append({"role": "assistant", "content": json.dumps({
                "action_type": action.action_type.value,
                "target":      action.target,
                "parameters":  {
                    k: (v.value if hasattr(v, "value") else v)
                    for k, v in action.parameters.items()
                },
            })})

            if done:
                break

    except Exception as exc:
        error_line = str(exc).replace("\n", " ").strip()
        print(
            f"[STEP] step={step + 1} action=no_op "
            f"reward=0.05 done=false error={error_line}",
            flush=True,
        )
        rewards.append(0.05)
        step += 1

    finally:
        if env is not None:
            env.close()

        if not rewards:
            rewards = [0.05]
            step    = 1

        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success={'true' if solved else 'false'} steps={step} "
            f"rewards={rewards_str}",
            flush=True,
        )


# ---------------- MAIN ----------------
def main() -> None:
    for task in list_tasks():
        _run_episode(task["id"])


if __name__ == "__main__":
    main()
