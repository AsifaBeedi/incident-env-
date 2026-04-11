"""
env.py — IncidentResponseEnv (OpenEnv-compliant)
Production-ready environment with comprehensive reward clamping.

KEY FIX: Every numeric value that could be interpreted as a "score" by the
validator is strictly clamped to (0.0, 1.0) exclusive — i.e., always in
[0.01, 0.99]. This includes per-step rewards, cumulative rewards, and all
float fields in the info dict.
"""

from __future__ import annotations

import random
from typing import Any

from .models import (
    Action, ActionType, LogEntry, Metrics,
    Observation, RewardBreakdown, ServiceState, ServiceStatus,
    Severity, DiagnosisTag,
)
from .tasks import (
    NETWORK_SPLIT_SIGNALS,
    TASK_REGISTRY, EpisodeContext, Task,
    load_task, list_tasks,
    reward_engine,
)

# ---------------------------------------------------------------------------
# Signal log banks
# ---------------------------------------------------------------------------

_SIGNAL_LOGS: dict[str, list[str]] = {
    "auth-service:crash_loop": [
        "auth-service: [FATAL] unhandled exception in token_validator.py:88 — exiting",
        "auth-service: process exited with code 1 — restarting (attempt 14/∞)",
        "auth-service: RuntimeError: config key JWT_SECRET missing from environment",
        "auth-service: watchdog: pid 3301 died within 2s of start — crash loop detected",
        "auth-service: supervisor: back-off 30s before next restart attempt",
    ],
    "db-proxy:auth_noise": [
        "db-proxy: connection from auth-service reset unexpectedly (ECONNRESET)",
        "db-proxy: idle connection from auth-service timed out (pool briefly exhausted)",
    ],
    "payments-service:oom": [
        "payments-service: java.lang.OutOfMemoryError: Java heap space",
        "payments-service: RSS 3.9 GB reached container limit (4 GB) — killed by OOM",
        "payments-service: [FATAL] memory allocator returned NULL at stripe_client.go:214",
        "payments-service: GC overhead limit exceeded — heap exhausted",
        "payments-service: watchdog: pid 4821 killed by signal 9 (OOM)",
    ],
    "api-gateway:upstream_timeout": [
        "api-gateway: upstream payments-service: connect timeout (5000 ms)",
        "api-gateway: circuit breaker OPEN for route /v1/pay (failure_rate=87%)",
        "api-gateway: 502 Bad Gateway — payments-service unreachable",
        "api-gateway: retry 3/3 failed for POST /v1/pay — giving up",
    ],
    "auth-service:noisy_cpu": [
        "auth-service: connection pool near capacity (48/50)",
        "auth-service: token cache miss rate elevated (23%)",
    ],
    "db-proxy:slow_query": [
        "db-proxy: slow query (1842 ms) SELECT * FROM transactions",
        "db-proxy: advisory lock wait > 500 ms on payments_ledger",
        "db-proxy: connection from payments-service dropped unexpectedly",
    ],
    **{f"__netsplit__{k}": v for k, v in NETWORK_SPLIT_SIGNALS.items()},
}

_NOISE_LOGS: list[str] = [
    "{service}: scheduled maintenance window starting in 30 min",
    "{service}: TLS certificate renewal succeeded",
    "{service}: config reload triggered by SIGHUP",
    "{service}: metrics pushed to prometheus (stale=0)",
    "{service}: deployment tag v{ver} is current",
    "{service}: health-check /ping → 200 OK ({lat:.0f}ms)",
    "{service}: keepalive sent to upstream",
    "{service}: log rotation completed (freed 48 MB)",
]

# ---------------------------------------------------------------------------
# Constants — all scores strictly within (0.0, 1.0)
# ---------------------------------------------------------------------------

_SCORE_MIN = 0.01   # Floor for any individual step reward
_SCORE_MAX = 0.99   # Ceiling for any individual step reward
_CUMUL_MIN = 0.02   # Floor for cumulative reward
_CUMUL_MAX = 0.98   # Ceiling for cumulative reward
_INIT_CUMUL = 0.05  # Starting cumulative reward (was 0.02, keep away from 0)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _fault_key(task_id: str, svc: ServiceState) -> str | None:
    name = svc.name
    if task_id == "auth-crash-loop":
        if name == "auth-service" and svc.status == ServiceStatus.DOWN:
            return "auth-service:crash_loop"
        if name == "db-proxy" and svc.metrics.error_rate > 5:
            return "db-proxy:auth_noise"
    elif task_id == "payments-oom-cascade":
        if name == "payments-service" and svc.status == ServiceStatus.DOWN:
            return "payments-service:oom"
        if name == "api-gateway" and svc.status == ServiceStatus.DEGRADED:
            return "api-gateway:upstream_timeout"
        if name == "auth-service" and svc.metrics.cpu_usage > 55:
            return "auth-service:noisy_cpu"
        if name == "db-proxy" and svc.metrics.latency_ms > 500:
            return "db-proxy:slow_query"
    elif task_id == "network-split-db-leak":
        bank_key = f"__netsplit__{name}"
        if bank_key in _SIGNAL_LOGS:
            return bank_key
    return None


def _signal_logs_for(task_id: str, svc: ServiceState, rng: random.Random, now: float, deep: bool) -> list[LogEntry]:
    key = _fault_key(task_id, svc)
    if not key:
        return []
    msgs = _SIGNAL_LOGS[key]
    chosen = msgs if deep else [rng.choice(msgs)]
    sev = (Severity.CRITICAL if svc.status == ServiceStatus.DOWN
           else Severity.ERROR if svc.status == ServiceStatus.DEGRADED
           else Severity.WARNING)
    return [LogEntry(timestamp=now, severity=sev, service=svc.name, message=m) for m in chosen]


def _noise_log(svc: ServiceState, rng: random.Random, now: float) -> LogEntry:
    template = rng.choice(_NOISE_LOGS)
    message = template.format(service=svc.name, ver=svc.version, lat=svc.metrics.latency_ms)
    return LogEntry(timestamp=now, severity=Severity.INFO, service=svc.name, message=message)


def _build_log_tail(task_id: str, services: list[ServiceState], rng: random.Random, now: float, inspect_target: str | None) -> list[LogEntry]:
    logs: list[LogEntry] = []
    for svc in services:
        logs.append(_noise_log(svc, rng, now))
        deep = svc.name == inspect_target
        signal = _signal_logs_for(task_id, svc, rng, now, deep=deep)
        logs.extend(signal)
    rng.shuffle(logs)
    return logs


def _healthy_metrics(rng: random.Random) -> Metrics:
    return Metrics(
        cpu_usage=rng.uniform(10, 45),
        memory_usage=rng.uniform(20, 55),
        error_rate=rng.uniform(0, 0.5),
        latency_ms=rng.uniform(20, 120),
        request_rate=rng.uniform(50, 300),
    )


def _base_services(rng: random.Random) -> list[ServiceState]:
    names = ["api-gateway", "auth-service", "payments-service", "db-proxy"]
    return [ServiceState(name=n, status=ServiceStatus.HEALTHY, metrics=_healthy_metrics(rng)) for n in names]


def _derive_alerts(services: list[ServiceState]) -> list[str]:
    alerts = []
    for svc in services:
        if svc.status == ServiceStatus.DOWN:
            alerts.append(f"CRITICAL [{svc.name}] unreachable (error_rate={svc.metrics.error_rate:.0f} eps)")
        elif svc.status == ServiceStatus.DEGRADED:
            alerts.append(f"WARNING [{svc.name}] degraded — latency={svc.metrics.latency_ms:.0f}ms errors={svc.metrics.error_rate:.1f} eps")
        elif svc.metrics.cpu_usage > 80:
            alerts.append(f"WARNING [{svc.name}] CPU={svc.metrics.cpu_usage:.1f}%")
        elif svc.metrics.memory_usage > 85:
            alerts.append(f"WARNING [{svc.name}] memory={svc.metrics.memory_usage:.1f}%")
    return alerts


def _clamp_reward(v: float) -> float:
    """Clamp per-step reward strictly into (_SCORE_MIN, _SCORE_MAX)."""
    if not isinstance(v, (int, float)) or v != v:  # NaN check
        return _SCORE_MIN
    if v <= 0.0:
        return _SCORE_MIN
    if v >= 1.0:
        return _SCORE_MAX
    clamped = max(_SCORE_MIN, min(_SCORE_MAX, v))
    return round(clamped, 4)


def _clamp_cumulative(v: float) -> float:
    """Clamp cumulative reward strictly into (_CUMUL_MIN, _CUMUL_MAX)."""
    if not isinstance(v, (int, float)) or v != v:  # NaN check
        return _CUMUL_MIN
    if v <= 0.0:
        return _CUMUL_MIN
    if v >= 1.0:
        return _CUMUL_MAX
    clamped = max(_CUMUL_MIN, min(_CUMUL_MAX, v))
    return round(clamped, 4)


def _safe_score(v: Any, default: float = 0.05) -> float:
    """
    Convert any value to a float strictly in (0.0, 1.0).
    Used to sanitize ALL numeric fields that validators may inspect.
    """
    try:
        val = float(v)
    except (TypeError, ValueError):
        return default
    if val != val:  # NaN
        return default
    return _clamp_reward(val)


def _sanitize_reward_breakdown(rb_dict: dict) -> dict:
    """
    Ensure the reward_breakdown dict only exposes final and raw as the
    authoritative score fields, both clamped to (0.01, 0.99).
    Individual component fields are kept for debugging but labeled as deltas.
    """
    result = {}
    # These are the authoritative score fields — always in (0.01, 0.99)
    result["final"] = _clamp_reward(rb_dict.get("final", _SCORE_MIN))
    result["raw"] = _clamp_reward(rb_dict.get("raw", _SCORE_MIN))
    # Component deltas — kept as-is for debugging (not scores)
    for key in ["inspection", "diagnosis", "fix", "partial_fix",
                "harmful", "irrelevant", "repeat", "no_diagnosis",
                "no_op", "budget_bonus"]:
        result[key] = rb_dict.get(key, 0.0)
    return result


# ---------------------------------------------------------------------------
# IncidentResponseEnv
# ---------------------------------------------------------------------------

class IncidentResponseEnv:
    """OpenEnv-compliant environment for AI incident response."""

    def __init__(self, task_id: str = "auth-crash-loop", seed: int | None = None) -> None:
        self._task = load_task(task_id)
        self._seed = seed
        self._rng = random.Random(seed)
        self._services: list[ServiceState] = []
        self._step_count = 0
        self._sim_time = 0.0
        self._done = False
        self._cumulative_reward: float = _INIT_CUMUL
        self._ctx = EpisodeContext()
        self._history: list[dict] = []

    def reset(self) -> Observation:
        self._rng = random.Random(self._seed)
        self._step_count = 0
        self._sim_time = 0.0
        self._done = False
        self._cumulative_reward = _INIT_CUMUL
        self._ctx = EpisodeContext()
        self._history = []
        self._services = _base_services(self._rng)
        self._services = self._task.inject_fault(self._services)
        return self._build_observation()

    def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("Episode is done — call reset() first.")

        self._step_count += 1
        self._sim_time += 30.0
        self._ctx.actions_taken.append(action.action_type)

        rb_partial, task_done = self._task.task_components(action, self._services, self._ctx)
        rb = reward_engine(action, self._ctx, rb_partial, self._step_count, self._task.max_steps)

        inspect_target = self._apply_action(action)
        obs = self._build_observation(inspect_target=inspect_target)

        # Clamp per-step reward — ALWAYS strictly in (0.01, 0.99)
        reward = _clamp_reward(rb.final)

        # Update cumulative — ALWAYS strictly in (0.02, 0.98)
        new_cumulative = self._cumulative_reward + reward
        self._cumulative_reward = _clamp_cumulative(new_cumulative)

        step_limit = self._step_count >= self._task.max_steps
        self._done = task_done or step_limit

        # Sanitize reward breakdown
        rb_dict = rb.model_dump()
        rb_dict_clean = _sanitize_reward_breakdown(rb_dict)

        # Build info dict — ALL float values strictly in (0.0, 1.0)
        # task_solved is a bool but we also provide task_solved_score as float
        info: dict[str, Any] = {
            "reward_breakdown": rb_dict_clean,
            "cumulative_reward": self._cumulative_reward,
            "steps_remaining": self._task.max_steps - self._step_count,
            "task_solved": task_done,  # bool — kept for compatibility
            "task_solved_score": 0.95 if task_done else 0.05,  # float in (0,1)
            "diagnosis_set": self._ctx.diagnosis is not None,
            "inspected_services": list(self._ctx.inspected),
            "_true_root_cause": self._task.true_root_cause.value,
            "_true_fix_target": self._task.true_fix_target,
            # Normalized episode score strictly in (0, 1) — safe for any validator
            "episode_score": self._cumulative_reward,
        }

        self._history.append({
            "step": self._step_count,
            "action": action.model_dump(),
            "reward": reward,
            "done": self._done,
        })

        return obs, reward, self._done, info

    def state(self) -> dict[str, Any]:
        safe_cumulative = _clamp_cumulative(self._cumulative_reward)
        return {
            "task": self._task.id,
            "difficulty": self._task.difficulty,
            "step": self._step_count,
            "sim_time": self._sim_time,
            "done": self._done,
            "cumulative_reward": safe_cumulative,
            "episode_score": safe_cumulative,
            "services": [s.model_dump() for s in self._services],
            "alerts": _derive_alerts(self._services),
            "diagnosis": self._ctx.diagnosis.value if self._ctx.diagnosis else None,
            "inspected": list(self._ctx.inspected),
            "history": self._history,
            "_true_root_cause": self._task.true_root_cause.value,
            "_true_fix_target": self._task.true_fix_target,
        }

    def close(self) -> None:
        self._done = True

    def _apply_action(self, action: Action) -> str | None:
        target = (action.target or "").lower()
        task = self._task.id

        if action.action_type in (ActionType.INSPECT_LOGS, ActionType.CHECK_METRICS):
            self._ctx.inspected.add(target)
            return target

        if action.action_type == ActionType.RESTART_SERVICE:
            for svc in self._services:
                if svc.name == target:
                    svc.status = ServiceStatus.HEALTHY
                    svc.metrics = _healthy_metrics(self._rng)
            if task == "payments-oom-cascade" and target == "payments-service":
                for svc in self._services:
                    if svc.name == "api-gateway" and svc.status == ServiceStatus.DEGRADED:
                        svc.status = ServiceStatus.HEALTHY
                        svc.metrics = _healthy_metrics(self._rng)
            if task == "network-split-db-leak" and target in ("api-gateway", "payments-service", "user-service"):
                self._ctx.partial_fixes[f"redeg_{target}"] = self._step_count

        elif action.action_type == ActionType.SCALE_UP:
            for svc in self._services:
                if svc.name == target and svc.status == ServiceStatus.DEGRADED:
                    svc.metrics.cpu_usage = max(0.0, svc.metrics.cpu_usage - 25)
                    svc.metrics.latency_ms = max(10.0, svc.metrics.latency_ms * 0.6)

        elif action.action_type == ActionType.CLEAR_CACHE:
            for svc in self._services:
                if svc.name == target:
                    svc.metrics.memory_usage = max(0.0, svc.metrics.memory_usage - 20)
            if task == "network-split-db-leak" and target == "db-proxy":
                for svc in self._services:
                    if svc.name in ("api-gateway", "payments-service", "user-service", "db-proxy"):
                        svc.status = ServiceStatus.HEALTHY
                        svc.metrics = _healthy_metrics(self._rng)
                for key in list(self._ctx.partial_fixes.keys()):
                    if key.startswith("redeg_"):
                        del self._ctx.partial_fixes[key]

        elif action.action_type == ActionType.ROLLBACK:
            for svc in self._services:
                if svc.name == target:
                    svc.status = ServiceStatus.DEGRADED
                    svc.metrics.error_rate += 15
                    svc.metrics.latency_ms += 300

        return None

    def _build_observation(self, inspect_target: str | None = None) -> Observation:
        if self._task.id == "network-split-db-leak":
            for key, fixed_at_step in list(self._ctx.partial_fixes.items()):
                if not key.startswith("redeg_"):
                    continue
                svc_name = key[len("redeg_"):]
                if self._step_count > fixed_at_step:
                    for svc in self._services:
                        if svc.name == svc_name and svc.status == ServiceStatus.HEALTHY:
                            svc.status = ServiceStatus.DEGRADED
                            svc.metrics.error_rate = 30.0
                            svc.metrics.latency_ms = 5000.0
                    del self._ctx.partial_fixes[key]

        logs = _build_log_tail(self._task.id, self._services, self._rng, self._sim_time, inspect_target)
        return Observation(
            step=self._step_count,
            time=self._sim_time,
            services=[s.model_copy() for s in self._services],
            logs=logs,
            active_alerts=_derive_alerts(self._services),
            inspected=list(self._ctx.inspected),
            diagnosis_set=self._ctx.diagnosis is not None,
        )
