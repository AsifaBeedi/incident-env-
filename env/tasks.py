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

def reward_engine(
    action: Action,
    ctx: EpisodeContext,
    task_rb: RewardBreakdown,
    step_count: int,
    max_steps: int,
) -> RewardBreakdown:
    rb = task_rb
    atype = action.action_type
    target = (action.target or "").lower()

    # NO NEGATIVE PENALTIES AT ALL. Replace all penalties with tiny positive survival points.
    if _seen(ctx, atype, target): rb.repeat = 0.01
    if atype in REMEDIATION_ACTIONS and ctx.diagnosis is None: rb.no_diagnosis = 0.01
    if atype == ActionType.NO_OP: rb.no_op = 0.01

    rb.budget_bonus = 0.05 if step_count == 1 else 0.01

    rb.raw = round(
        rb.inspection + rb.diagnosis + rb.fix + rb.partial_fix +
        rb.harmful + rb.irrelevant + rb.repeat + rb.no_diagnosis +
        rb.no_op + rb.budget_bonus, 4
    )
    rb.final = round(max(0.01, min(0.99, rb.raw)), 4)
    return rb

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
# TASK 1
# ===========================================================================

def _inject_auth_crash(services: list[ServiceState]) -> list[ServiceState]:
    for svc in services:
        if svc.name == "auth-service":
            svc.status  = ServiceStatus.DOWN
            svc.metrics = Metrics(cpu_usage=0.0, memory_usage=12.0, error_rate=180.0, latency_ms=0.0, request_rate=0.0)
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
    atype, target = action.action_type, (action.target or "").lower()
    diag = action.parameters.get("diagnosis")
    
    rb.budget_bonus = 0.01 # baseline survival point

    if diag and not ctx.diagnosis:
        ctx.diagnosis = diag
        rb.diagnosis = 0.10 if diag == DiagnosisTag.CRASH_LOOP else 0.01

    if atype in (ActionType.INSPECT_LOGS, ActionType.CHECK_METRICS):
        rb.inspection = 0.10 if (target == "auth-service" and target not in ctx.inspected) else 0.01
        ctx.inspected.add(target)
        return rb, done

    if atype == ActionType.ACKNOWLEDGE:
        rb.inspection = 0.01
        return rb, done

    if atype == ActionType.RESTART_SERVICE:
        if target == "auth-service": rb.fix, done = 0.20, True
        else: rb.irrelevant = 0.01
        ctx.remediation_count += 1
        return rb, done

    if atype in (ActionType.ROLLBACK, ActionType.CLEAR_CACHE):
        rb.harmful = 0.01
        ctx.remediation_count += 1
        return rb, done

    if atype == ActionType.SCALE_UP:
        rb.irrelevant = 0.01
        ctx.remediation_count += 1
        return rb, done

    return rb, done

TASK_AUTH_CRASH = Task(
    id="auth-crash-loop", name="Auth Service Crash Loop", description=".", difficulty="easy", max_steps=6,
    inject_fault=_inject_auth_crash, task_components=_components_auth_crash,
    true_root_cause=DiagnosisTag.CRASH_LOOP, true_fix_target="auth-service",
)

# ===========================================================================
# TASK 2
# ===========================================================================

def _inject_payments_oom(services: list[ServiceState]) -> list[ServiceState]:
    for svc in services:
        if svc.name == "payments-service":
            svc.status, svc.metrics = ServiceStatus.DOWN, Metrics(cpu_usage=0.0, memory_usage=99.8, error_rate=320.0, latency_ms=0.0, request_rate=0.0)
        elif svc.name == "api-gateway":
            svc.status, svc.metrics = ServiceStatus.DEGRADED, Metrics(cpu_usage=72.0, memory_usage=58.0, error_rate=45.0, latency_ms=4200.0, request_rate=180.0)
        elif svc.name == "auth-service":
            svc.metrics.cpu_usage, svc.metrics.latency_ms = 61.0, 310.0
        elif svc.name == "db-proxy":
            svc.metrics.latency_ms = 850.0
    return services

def _components_payments_oom(
    action: Action,
    services: list[ServiceState],
    ctx: EpisodeContext,
) -> tuple[RewardBreakdown, bool]:
    rb, done = RewardBreakdown(), False
    atype, target = action.action_type, (action.target or "").lower()
    diag = action.parameters.get("diagnosis")
    
    rb.budget_bonus = 0.01

    if diag and not ctx.diagnosis:
        ctx.diagnosis = diag
        rb.diagnosis = 0.10 if diag == DiagnosisTag.OOM_KILL else 0.01

    if atype in (ActionType.INSPECT_LOGS, ActionType.CHECK_METRICS):
        if target == "payments-service" and target not in ctx.inspected: rb.inspection = 0.10
        elif target == "api-gateway" and target not in ctx.inspected: rb.inspection = 0.05
        else: rb.inspection = 0.01
        ctx.inspected.add(target)
        return rb, done

    if atype == ActionType.ACKNOWLEDGE:
        rb.inspection = 0.01
        return rb, done

    if atype == ActionType.RESTART_SERVICE:
        if target == "payments-service": rb.fix, done = 0.20, True
        elif target == "api-gateway": rb.partial_fix = 0.05
        else: rb.irrelevant = 0.01
        ctx.remediation_count += 1
        return rb, done

    if atype in (ActionType.ROLLBACK, ActionType.CLEAR_CACHE, ActionType.SCALE_UP):
        rb.harmful = 0.01
        ctx.remediation_count += 1
        return rb, done

    return rb, done

TASK_PAYMENTS_OOM = Task(
    id="payments-oom-cascade", name="Payments OOM Cascade", description=".", difficulty="medium", max_steps=8,
    inject_fault=_inject_payments_oom, task_components=_components_payments_oom,
    true_root_cause=DiagnosisTag.OOM_KILL, true_fix_target="payments-service",
)

# ===========================================================================
# TASK 3
# ===========================================================================

def _inject_network_split(services: list[ServiceState]) -> list[ServiceState]:
    from .models import ServiceState as S, ServiceStatus as SS
    services.append(S(name="user-service", status=SS.DEGRADED, metrics=Metrics(cpu_usage=44.0, memory_usage=51.0, error_rate=9.0, latency_ms=3100.0, request_rate=88.0), version="2.3.1"))
    for svc in services:
        if svc.name in ("db-proxy", "api-gateway", "payments-service"):
            svc.status = ServiceStatus.DEGRADED
            svc.metrics = Metrics(cpu_usage=82.0, memory_usage=78.0, error_rate=18.0, latency_ms=8400.0, request_rate=95.0)
    return services

def _components_network_split(
    action: Action,
    services: list[ServiceState],
    ctx: EpisodeContext,
) -> tuple[RewardBreakdown, bool]:
    rb, done = RewardBreakdown(), False
    atype, target = action.action_type, (action.target or "").lower()
    diag = action.parameters.get("diagnosis")
    
    rb.budget_bonus = 0.01

    if diag and not ctx.diagnosis:
        ctx.diagnosis = diag
        rb.diagnosis = 0.10 if diag == DiagnosisTag.RESOURCE_SATURATION else 0.01

    if atype in (ActionType.INSPECT_LOGS, ActionType.CHECK_METRICS):
        if target == "db-proxy" and target not in ctx.inspected: rb.inspection = 0.10
        elif target in ("api-gateway", "payments-service", "user-service") and target not in ctx.inspected: rb.inspection = 0.05
        else: rb.inspection = 0.01
        ctx.inspected.add(target)
        return rb, done

    if atype == ActionType.CLEAR_CACHE:
        if target == "db-proxy":
            rb.fix, done = 0.15, True
        ctx.remediation_count += 1
        return rb, done

    if atype == ActionType.RESTART_SERVICE:
        if target in ("api-gateway", "payments-service", "user-service", "db-proxy"):
            rb.partial_fix = 0.02
        ctx.remediation_count += 1
        return rb, done

    return rb, done

TASK_NETWORK_SPLIT = Task(
    id="network-split-db-leak",
    name="Network Split — DB WAL Saturation",
    description="Multiple services are degraded with three plausible root causes.",
    difficulty="hard",
    max_steps=10,
    inject_fault=_inject_network_split,
    task_components=_components_network_split,
    true_root_cause=DiagnosisTag.RESOURCE_SATURATION,
    true_fix_target="db-proxy",
)

# ===========================================================================
# Registry
# ===========================================================================

TASK_REGISTRY: dict[str, Task] = {
    TASK_AUTH_CRASH.id:    TASK_AUTH_CRASH,
    TASK_PAYMENTS_OOM.id:  TASK_PAYMENTS_OOM,
    TASK_NETWORK_SPLIT.id: TASK_NETWORK_SPLIT,
}

def load_task(task_id: str) -> Task:
    return TASK_REGISTRY[task_id]

def list_tasks() -> list[dict]:
    return [{"id": t.id, "name": t.name, "difficulty": t.difficulty, "max_steps": t.max_steps} for t in TASK_REGISTRY.values()]
