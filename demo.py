"""
demo.py — Exercises all three tasks and every reward rule.

Run:  python demo.py
"""

from __future__ import annotations
import json
from env import IncidentResponseEnv, Action, ActionType, DiagnosisTag, list_tasks

SEP = "─" * 68


def _rb(info: dict) -> str:
    rb = info["reward_breakdown"]
    active = {k: v for k, v in rb.items()
              if v != 0.0 and k not in ("raw", "final")}
    return f"components={active}  raw={rb['raw']}  final={rb['final']}"


def run(task_id: str, label: str, steps: list[Action]) -> float:
    print(f"\n{'═'*68}")
    print(f"  {label}")
    print(f"  task={task_id}")
    print("═"*68)

    env = IncidentResponseEnv(task_id=task_id, seed=42)
    obs = env.reset()

    print("  Alerts:")
    for a in obs.active_alerts:
        print(f"    {a}")

    cum = 0.0
    for i, action in enumerate(steps, 1):
        obs, reward, done, info = env.step(action)
        cum = info["cumulative_reward"]
        diag = action.parameters.get("diagnosis", "—")
        print(
            f"\n  step {i}  {action.action_type.value:22s} "
            f"target={action.target or '—':22s} diag={diag}"
        )
        print(f"         reward={reward:.4f}  {_rb(info)}")
        print(f"         done={done}  task_solved={info['task_solved']}")
        if done:
            break

    print(f"\n  ► cumulative={cum:.4f}")
    return cum


# ============================================================
def main():
    print(f"\n{SEP}\n  Registered tasks\n{SEP}")
    for t in list_tasks():
        print(f"  {t['id']:38s}  {t['difficulty']:6s}  max_steps={t['max_steps']}")

    # ── EASY ───────────────────────────────────────────────────────────
    run("auth-crash-loop", "EASY — optimal: inspect → diagnose → fix", [
        Action(action_type=ActionType.INSPECT_LOGS, target="auth-service"),
        Action(action_type=ActionType.RESTART_SERVICE, target="auth-service",
               parameters={"diagnosis": DiagnosisTag.CRASH_LOOP}),
    ])

    run("auth-crash-loop", "EASY — penalty: fix without diagnosis", [
        Action(action_type=ActionType.RESTART_SERVICE, target="auth-service"),
    ])

    run("auth-crash-loop", "EASY — penalty: wrong diagnosis + harmful action", [
        Action(action_type=ActionType.INSPECT_LOGS, target="db-proxy"),
        Action(action_type=ActionType.ROLLBACK, target="auth-service",
               parameters={"diagnosis": DiagnosisTag.BAD_DEPLOY}),
    ])

    run("auth-crash-loop", "EASY — penalty: repeat action", [
        Action(action_type=ActionType.INSPECT_LOGS, target="auth-service"),
        Action(action_type=ActionType.INSPECT_LOGS, target="auth-service"),  # repeat
        Action(action_type=ActionType.RESTART_SERVICE, target="auth-service",
               parameters={"diagnosis": DiagnosisTag.CRASH_LOOP}),
    ])

    # ── MEDIUM ─────────────────────────────────────────────────────────
    run("payments-oom-cascade", "MEDIUM — optimal: inspect both + diagnose + fix", [
        Action(action_type=ActionType.INSPECT_LOGS,  target="payments-service"),
        Action(action_type=ActionType.CHECK_METRICS, target="api-gateway"),
        Action(action_type=ActionType.RESTART_SERVICE, target="payments-service",
               parameters={"diagnosis": DiagnosisTag.OOM_KILL}),
    ])

    run("payments-oom-cascade", "MEDIUM — partial fix then correct", [
        Action(action_type=ActionType.RESTART_SERVICE, target="api-gateway"),  # partial
        Action(action_type=ActionType.INSPECT_LOGS,    target="payments-service"),
        Action(action_type=ActionType.RESTART_SERVICE, target="payments-service",
               parameters={"diagnosis": DiagnosisTag.OOM_KILL}),
    ])

    run("payments-oom-cascade", "MEDIUM — wrong diagnosis (upstream timeout)", [
        Action(action_type=ActionType.INSPECT_LOGS,    target="db-proxy"),
        Action(action_type=ActionType.RESTART_SERVICE, target="payments-service",
               parameters={"diagnosis": DiagnosisTag.UPSTREAM_TIMEOUT}),
    ])

    # ── HARD ───────────────────────────────────────────────────────────
    run("network-split-db-leak", "HARD — optimal: full investigation → correct fix", [
        Action(action_type=ActionType.INSPECT_LOGS,    target="db-proxy"),
        Action(action_type=ActionType.CHECK_METRICS,   target="payments-service"),
        Action(action_type=ActionType.INSPECT_LOGS,    target="api-gateway"),
        Action(action_type=ActionType.CLEAR_CACHE,     target="db-proxy",
               parameters={"diagnosis": DiagnosisTag.RESOURCE_SATURATION}),
    ])

    run("network-split-db-leak", "HARD — mislead: restart frontend (delayed re-degrade)", [
        Action(action_type=ActionType.RESTART_SERVICE, target="api-gateway"),
        # Next obs: api-gateway appears healthy (false signal)
        Action(action_type=ActionType.RESTART_SERVICE, target="payments-service"),
        # Next obs: payments appears healthy (false signal), then re-degrades
        Action(action_type=ActionType.NO_OP),
        # Both services degrade again — agent wasted 3 steps
        Action(action_type=ActionType.INSPECT_LOGS, target="db-proxy"),
        Action(action_type=ActionType.CLEAR_CACHE,  target="db-proxy",
               parameters={"diagnosis": DiagnosisTag.RESOURCE_SATURATION}),
    ])

    run("network-split-db-leak", "HARD — wrong hypothesis: OOM on payments", [
        Action(action_type=ActionType.INSPECT_LOGS,    target="payments-service"),
        Action(action_type=ActionType.RESTART_SERVICE, target="payments-service",
               parameters={"diagnosis": DiagnosisTag.OOM_KILL}),
        # payments briefly recovers, then re-degrades next step
        Action(action_type=ActionType.NO_OP),
        Action(action_type=ActionType.NO_OP),
    ])

    run("network-split-db-leak", "HARD — wrong hypothesis: bad deploy on gateway", [
        Action(action_type=ActionType.INSPECT_LOGS, target="api-gateway"),
        Action(action_type=ActionType.ROLLBACK,     target="api-gateway",
               parameters={"diagnosis": DiagnosisTag.BAD_DEPLOY}),
    ])

    # ── Reward rule summary ────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Reward rule verification")
    print(SEP)
    print("  Rule 1  repeat action           → repeat=-0.1 in breakdown")
    print("  Rule 2  fix without diagnosis   → no_diagnosis=-0.2 in breakdown")
    print("  Rule 3  correct sequence        → inspection=+0.2, diagnosis=+0.3, fix=+0.5")
    print("  Rule 4  wrong diagnosis         → diagnosis=-0.3 (or -0.1 if plausible)")
    print("  Rule 5a irrelevant action       → irrelevant=-0.1")
    print("  Rule 5b harmful action          → harmful=-0.3")
    print("  Rule 6  all rewards clamped     → final ∈ [0.0, 1.0]")
    print("  Rule 7  breakdown in info       → info['reward_breakdown']")


if __name__ == "__main__":
    main()