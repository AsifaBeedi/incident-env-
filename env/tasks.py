"""
tasks.py — compatible with env.py exactly as written.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .models import (
    Action, ActionType, DiagnosisTag,
    Metrics, RewardBreakdown, REMEDIATION_ACTIONS,
    ServiceState, ServiceStatus,
)

# ---------------------------------------------------------------------------
# EpisodeContext
# ---------------------------------------------------------------------------

@dataclass
class EpisodeContext:
    inspected:         set[str]            = field(default_factory=set)
    diagnosis:         DiagnosisTag | None = None
    actions_taken:     list[ActionType]    = field(default_factory=list)
    remediation_count: int                 = 0
    partial_fixes:     dict[str, int]      = field(default_factory=dict)
    _action_pairs:     set[tuple]          = field(default_factory=set, repr=False)

def _seen(ctx: EpisodeContext, atype: ActionType, target: str) -> bool:
    pair = (atype, target)
    already = pair in ctx._action_pairs
    ctx._action_pairs.add(pair)
    return already

# ---------------------------------------------------------------------------
# Reward engine - CRITICAL FIX
# ---------------------------------------------------------------------------

def reward_engine(
    action: Action,
    ctx: EpisodeContext,
    task_rb: RewardBreakdown,
    step_count: int,
    max_steps: int,
) -> RewardBreakdown:
    rb     = task_rb
    atype  = action.action_type
    target = (action.target or "").lower()

    if _seen(ctx, atype, target):
        rb.repeat = -0.08  # Changed from -0.1

    if atype in REMEDIATION_ACTIONS and ctx.diagnosis is None:
        rb.no_diagnosis = -0.15  # Changed from -0.2

    if atype == ActionType.NO_OP:
        rb.no_op = -0.03  # Changed from -0.05

    pre_clamp = (rb.inspection + rb.diagnosis + rb.fix + rb.partial_fix
                 + rb.harmful + rb.irrelevant)
                 
    if pre_clamp > 0 and step_count <= max_steps // 2:
        rb.budget_bonus = 0.03  # Changed from 0.05

    if step_count == 1:
        rb.budget_bonus += 0.10  # Changed from 0.15

    rb.raw = round(
        rb.inspection + rb.diagnosis + rb.fix + rb.partial_fix
        + rb.harmful + rb.irrelevant
        + rb.repeat + rb.no_diagnosis + rb.no_op + rb.budget_bonus,
        4,
    )
    
    # 🔥 CRITICAL FIX: Use 0.02 minimum and 0.98 maximum
    rb.final = round(max(0.02, min(0.98, rb.raw)), 4)
    return rb

# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------

@dataclass
class Task:
    id:              str
    name:            str
    description:     str
    difficulty:      str
    max_steps:       int
    inject_fault:    Callable[[list[ServiceState]], list[ServiceState]] = field(repr=False)
    task_components: Callable[
        [Action, list[ServiceState], EpisodeContext],
        tuple[RewardBreakdown, bool],
    ] = field(repr=False)
    true_root_cause: DiagnosisTag = DiagnosisTag.CRASH_LOOP
    true_fix_target: str          = ""

# ===========================================================================
# TASK 1 — EASY  auth-crash-loop
# ===========================================================================

def _inject_auth_crash(services: list[ServiceState]) -> list[ServiceState]:
    for svc in services:
        if svc.name == "auth-service":
            svc.status  = ServiceStatus.DOWN
            svc.metrics = Metrics(
                cpu_usage=0.0, memory_usage=12.0,
                error_rate=180.0, latency_ms=0.0, request_rate=0.0,
            )
        elif svc.name == "db-proxy":
            svc.metrics.error_rate = 8.0
            svc.metrics.latency_ms = 420.0
    return services


def _components_auth_crash(
    action: Action,
    services: list[ServiceState],
    ctx: EpisodeContext,
) -> tuple[RewardBreakdown, bool]:
    rb, done = RewardBreakdown(), False
    atype  = action.action_type
    target = (action.target or "").lower()
    diag   = action.parameters.get("diagnosis")

    if diag and not ctx.diagnosis:
        ctx.diagnosis = diag
        rb.diagnosis  = 0.15 if diag == DiagnosisTag.CRASH_LOOP else -0.08  # Changed

    if atype in (ActionType.INSPECT_LOGS, ActionType.CHECK_METRICS):
        rb.inspection = 0.15 if (target == "auth-service" and target not in ctx.inspected) \
                        else (0.03 if target and target not in ctx.inspected else 0.0)  # Changed
        ctx.inspected.add(target)
        return rb, done

    if atype == ActionType.ACKNOWLEDGE:
        rb.inspection = 0.0
        return rb, done

    if atype == ActionType.RESTART_SERVICE:
        if target == "auth-service":
            rb.fix, done = 0.25, True  # Changed from 0.20
        else:
            rb.irrelevant = -0.08  # Changed
        ctx.remediation_count += 1
        return rb, done

    if atype in (ActionType.ROLLBACK, ActionType.CLEAR_CACHE):
        rb.harmful = -0.15  # Changed
        ctx.remediation_count += 1
        return rb, done

    if atype == ActionType.SCALE_UP:
        rb.irrelevant = -0.08  # Changed
        ctx.remediation_count += 1
        return rb, done

    return rb, done

TASK_AUTH_CRASH = Task(
    id="auth-crash-loop",
    name="Auth Service Crash Loop",
    description="One or more services are reporting errors. Identify the root cause and restore normal operation.",
    difficulty="easy",
    max_steps=6,
    inject_fault=_inject_auth_crash,
    task_components=_components_auth_crash,
    true_root_cause=DiagnosisTag.CRASH_LOOP,
    true_fix_target="auth-service",
)

# ===========================================================================
# TASK 2 — MEDIUM  payments-oom-cascade
# ===========================================================================

def _inject_payments_oom(services: list[ServiceState]) -> list[ServiceState]:
    for svc in services:
        if svc.name == "payments-service":
            svc.status  = ServiceStatus.DOWN
            svc.metrics = Metrics(
                cpu_usage=0.0, memory_usage=99.8,
                error_rate=320.0, latency_ms=0.0, request_rate=0.0,
            )
        elif svc.name == "api-gateway":
            svc.status  = ServiceStatus.DEGRADED
            svc.metrics = Metrics(
                cpu_usage=72.0, memory_usage=58.0,
                error_rate=45.0, latency_ms=4200.0, request_rate=180.0,
            )
        elif svc.name == "auth-service":
            svc.metrics.cpu_usage  = 61.0
            svc.metrics.latency_ms = 310.0
        elif svc.name == "db-proxy":
            svc.metrics.latency_ms = 850.0
    return services

def _components_payments_oom(
    action: Action,
    services: list[ServiceState],
    ctx: EpisodeContext,
) -> tuple[RewardBreakdown, bool]:
    rb, done = RewardBreakdown(), False
    atype  = action.action_type
    target = (action.target or "").lower()
    diag   = action.parameters.get("diagnosis")

    if diag and not ctx.diagnosis:
        ctx.diagnosis = diag
        if diag == DiagnosisTag.OOM_KILL:
            rb.diagnosis = 0.15  # Changed
        elif diag == DiagnosisTag.UPSTREAM_TIMEOUT:
            rb.diagnosis = -0.08  # Changed
        else:
            rb.diagnosis = -0.15  # Changed

    if atype in (ActionType.INSPECT_LOGS, ActionType.CHECK_METRICS):
        if target == "payments-service" and target not in ctx.inspected:
            rb.inspection = 0.15  # Changed
        elif target == "api-gateway" and target not in ctx.inspected:
            rb.inspection = 0.05  # Changed
        elif target and target not in ctx.inspected:
            rb.inspection = 0.03  # Changed
        ctx.inspected.add(target)
        return rb, done

    if atype == ActionType.ACKNOWLEDGE:
        rb.inspection = 0.0
        return rb, done

    if atype == ActionType.RESTART_SERVICE:
        if target == "payments-service":
            rb.fix, done = 0.25, True  # Changed
        elif target == "api-gateway":
            rb.partial_fix = 0.05  # Changed
        else:
            rb.irrelevant = -0.08  # Changed
        ctx.remediation_count += 1
        return rb, done

    if atype == ActionType.ROLLBACK:
        rb.harmful = -0.15  # Changed
        ctx.remediation_count += 1
        return rb, done

    if atype in (ActionType.SCALE_UP, ActionType.CLEAR_CACHE):
        rb.irrelevant = -0.08  # Changed
        ctx.remediation_count += 1
        return rb, done

    return rb, done

TASK_PAYMENTS_OOM = Task(
    id="payments-oom-cascade",
    name="Payments OOM Cascade",
    description="Multiple services are showing anomalies. Identify the true root cause and apply the correct fix.",
    difficulty="medium",
    max_steps=8,
    inject_fault=_inject_payments_oom,
    task_components=_components_payments_oom,
    true_root_cause=DiagnosisTag.OOM_KILL,
    true_fix_target="payments-service",
)

# ===========================================================================
# TASK 3 — HARD  network-split-db-leak
# ===========================================================================

NETWORK_SPLIT_SIGNALS: dict[str, list[str]] = {
    "db-proxy": [
        "db-proxy: pg_wal directory size 47 GB — approaching disk limit (50 GB)",
        "db-proxy: standby replica lag: ∞ (replica unreachable since 04:12 UTC)",
        "db-proxy: WARNING checkpoint completion taking >30s — WAL write stall",
        "db-proxy: write latency P99 = 8412 ms (threshold: 500 ms)",
        "db-proxy: shared_buffers usage 74% — consider increasing",
    ],
    "api-gateway": [
        "api-gateway: upstream db-proxy: write transaction timeout after 6000ms",
        "api-gateway: POST /v1/checkout → 503 Service Unavailable (db write failed)",
        "api-gateway: WARNING version mismatch detected in /health response headers",
        "api-gateway: last deployment: 6h ago (tag v3.1.4) — rollback available",
    ],
    "payments-service": [
        "payments-service: DB write timeout after 7800ms for txn_id=TX991023",
        "payments-service: heap memory 62% — growth rate 0.3%/min (non-linear)",
        "payments-service: WARNING memory growth detected — monitoring",
    ],
    "user-service": [
        "user-service: SELECT latency 3100ms — lock contention on payments table",
        "user-service: read query queued 12s waiting for lock release from db-proxy",
    ],
    "auth-service": [
        "auth-service: memory usage 81% — GC pressure from session cache growth",
        "auth-service: INFO read replica shard healthy — no issues detected",
    ],
}

def _inject_network_split(services: list[ServiceState]) -> list[ServiceState]:
    from .models import ServiceState as S, ServiceStatus as SS

    services.append(S(
        name="user-service", status=SS.DEGRADED,
        metrics=Metrics(cpu_usage=44.0, memory_usage=51.0,
                        error_rate=9.0, latency_ms=3100.0, request_rate=88.0),
        version="2.3.1",
    ))

    for svc in services:
        if svc.name == "db-proxy":
            svc.status  = ServiceStatus.DEGRADED
            svc.metrics = Metrics(
                cpu_usage=82.0, memory_usage=78.0,
                error_rate=18.0, latency_ms=8400.0, request_rate=95.0,
            )
        elif svc.name == "api-gateway":
            svc.status  = ServiceStatus.DEGRADED
            svc.metrics = Metrics(
                cpu_usage=55.0, memory_usage=47.0,
                error_rate=28.0, latency_ms=6100.0, request_rate=195.0,
            )
        elif svc.name == "payments-service":
            svc.status  = ServiceStatus.DEGRADED
            svc.metrics = Metrics(
                cpu_usage=68.0, memory_usage=62.0,
                error_rate=41.0, latency_ms=7800.0, request_rate=120.0,
            )
        elif svc.name == "auth-service":
            svc.metrics.memory_usage = 81.0
            svc.metrics.latency_ms   = 145.0
    return services

def _components_network_split(
    action: Action,
    services: list[ServiceState],
    ctx: EpisodeContext,
) -> tuple[RewardBreakdown, bool]:
    rb, done = RewardBreakdown(), False
    atype  = action.action_type
    target = (action.target or "").lower()
    diag   = action.parameters.get("diagnosis")

    if diag and not ctx.diagnosis:
        ctx.diagnosis = diag
        if diag == DiagnosisTag.RESOURCE_SATURATION:
            rb.diagnosis = 0.15  # Changed
        elif diag in (DiagnosisTag.OOM_KILL, DiagnosisTag.BAD_DEPLOY):
            rb.diagnosis = -0.08  # Changed
        else:
            rb.diagnosis = -0.15  # Changed

    if atype in (ActionType.INSPECT_LOGS, ActionType.CHECK_METRICS):
        if target == "db-proxy" and target not in ctx.inspected:
            rb.inspection = 0.15  # Changed
        elif target in ("api-gateway", "payments-service", "user-service") \
                and target not in ctx.inspected:
            rb.inspection = 0.03  # Changed
        elif target not in ctx.inspected:
            rb.inspection = 0.02  # Changed
        ctx.inspected.add(target)
        return rb, done

    if atype == ActionType.ACKNOWLEDGE:
        rb.inspection = 0.0
        return rb, done

    if atype == ActionType.CLEAR_CACHE:
        if target == "db-proxy":
            db_inspected  = "db-proxy" in ctx.inspected
            corroborated  = bool(ctx.inspected & {"api-gateway", "payments-service", "user-service"})
            rb.fix        = 0.25 if (db_inspected and corroborated) else (0.18 if db_inspected else 0.08)  # Changed
            done          = True
        else:
            rb.irrelevant = -0.08  # Changed
        ctx.remediation_count += 1
        return rb, done

    if atype == ActionType.RESTART_SERVICE:
        if target in ("api-gateway", "payments-service", "user-service"):
            ctx.partial_fixes[target] = ctx.partial_fixes.get(target, 0) + 1
            rb.partial_fix = 0.03  # Changed
        elif target == "db-proxy":
            rb.partial_fix = 0.03  # Changed
        elif target == "auth-service":
            rb.harmful = -0.15  # Changed
        else:
            rb.irrelevant = -0.08  # Changed
        ctx.remediation_count += 1
        return rb, done

    if atype == ActionType.SCALE_UP:
        rb.partial_fix = 0.02 if target == "db-proxy" else 0.0  # Changed
        ctx.remediation_count += 1
        return rb, done

    if atype == ActionType.ROLLBACK:
        rb.harmful = -0.15  # Changed
        ctx.remediation_count += 1
        return rb, done

    return rb, done

TASK_NETWORK_SPLIT = Task(
    id="network-split-db-leak",
    name="Network Split — DB WAL Saturation",
    description="Multiple services are degraded with three plausible root causes. Investigate carefully and apply the correct fix.",
    difficulty="hard",
    max_steps=10,
    inject_fault=_inject_network_split,
    task_components=_components_network_split,
    true_root_cause=DiagnosisTag.RESOURCE_SATURATION,
    true_fix_target="db-proxy",
)

# ===========================================================================
# Registry + loader
# ===========================================================================

TASK_REGISTRY: dict[str, Task] = {
    TASK_AUTH_CRASH.id:    TASK_AUTH_CRASH,
    TASK_PAYMENTS_OOM.id:  TASK_PAYMENTS_OOM,
    TASK_NETWORK_SPLIT.id: TASK_NETWORK_SPLIT,
}

def load_task(task_id: str) -> Task:
    if task_id not in TASK_REGISTRY:
        available = ", ".join(f"{t.id!r} ({t.difficulty})" for t in TASK_REGISTRY.values())
        raise ValueError(f"Unknown task {task_id!r}. Available: {available}")
    return TASK_REGISTRY[task_id]

def list_tasks() -> list[dict]:
    return [
        {"id": t.id, "name": t.name, "difficulty": t.difficulty, "max_steps": t.max_steps}
        for t in TASK_REGISTRY.values()
        ]
