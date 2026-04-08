"""
tasks.py — Task definitions and the shared reward engine.

Architecture
------------
  Each task provides two pure functions:

    inject_fault(services) → services
        Mutates the baseline service list to create the broken initial state.

    task_components(action, services, ctx) → RewardBreakdown
        Fills in the task-specific reward components (inspection, diagnosis,
        fix, partial_fix, harmful, irrelevant) WITHOUT touching penalties that
        the engine owns (repeat, no_diagnosis, no_op, budget_bonus).
        Returns a partially-filled RewardBreakdown.

  The shared reward_engine() in this file applies universal rules on top:
    - repeat-action penalty  (-0.1)
    - no-diagnosis penalty   (-0.2, if remediation attempted without diagnosis)
    - clamp to [0.0, 1.0]

  This separation means: task scorers never re-implement the same penalty logic,
  and adding a new universal rule only requires changing one place.

Tasks
-----
  easy   auth-crash-loop          1–2 step fix; single obvious fault
  medium payments-oom-cascade     2–3 step fix; downstream confusion
  hard   network-split-db-leak    3–4 step fix; three competing hypotheses,
                                  partial fixes that mislead, delayed symptoms
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
    _action_pairs:     set[tuple]          = field(default_factory=set, repr=False)
    # Hard-task specific: tracks partial-fix state for delayed symptom simulation
    partial_fixes:     dict[str, int]      = field(default_factory=dict)


def _seen(ctx: EpisodeContext, atype: ActionType, target: str) -> bool:
    """Check and register (action_type, target). Returns True if already seen."""
    pair = (atype, target)
    already = pair in ctx._action_pairs
    ctx._action_pairs.add(pair)
    return already


# ---------------------------------------------------------------------------
# Shared reward engine
# ---------------------------------------------------------------------------

def reward_engine(
    action: Action,
    ctx: EpisodeContext,
    task_rb: RewardBreakdown,
    step_count: int,
    max_steps: int,
) -> RewardBreakdown:
    """
    Apply universal penalty rules on top of a task-computed RewardBreakdown.

    Rules applied here (in order):
      1. Repeat-action penalty  : -0.1 if (action_type, target) already seen
      2. No-diagnosis penalty   : -0.2 if remediation attempted without diagnosis
      3. NO_OP penalty          : -0.05
      4. Budget bonus           : +0.05 if positive score in first half of episode
      5. Clamp raw → [0.0, 1.0]
    """
    rb     = task_rb
    target = (action.target or "").lower()
    atype  = action.action_type

    # Rule 1 — repeat penalty (engine checks here, after task scorer has run)
    if _seen(ctx, atype, target):
        rb.repeat = -0.1

    # Rule 2 — remediation without diagnosis
    if atype in REMEDIATION_ACTIONS and ctx.diagnosis is None:
        rb.no_diagnosis = -0.2

    # Rule 3 — NO_OP
    if atype == ActionType.NO_OP:
        rb.no_op = -0.05

    # Rule 4 — budget bonus (early correct action)
    pre_clamp = (rb.inspection + rb.diagnosis + rb.fix + rb.partial_fix
                 + rb.harmful + rb.irrelevant)
    if pre_clamp > 0 and step_count <= max_steps // 2:
        rb.budget_bonus = 0.05

    # Totals
    rb.raw   = round(
        rb.inspection + rb.diagnosis + rb.fix + rb.partial_fix
        + rb.harmful + rb.irrelevant
        + rb.repeat + rb.no_diagnosis + rb.no_op + rb.budget_bonus,
        4,
    )
    rb.final = round(max(0.0, min(1.0, rb.raw)), 4)
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
    # Returns (partially-filled RewardBreakdown, episode_done)
    task_components: Callable[
        [Action, list[ServiceState], EpisodeContext],
        tuple[RewardBreakdown, bool],
    ] = field(repr=False)
    true_root_cause: DiagnosisTag = DiagnosisTag.CRASH_LOOP
    true_fix_target: str          = ""


# ===========================================================================
# TASK 1 — EASY  "auth-crash-loop"
# ===========================================================================
# Root cause : auth-service is in a crash loop (missing JWT_SECRET env var).
# Red herring: db-proxy shows connection errors because auth keeps resetting
#              connections — looks like a DB issue at first glance.
# Correct seq: INSPECT_LOGS auth-service
#              → RESTART_SERVICE auth-service (diagnosis=CRASH_LOOP)
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
    rb     = RewardBreakdown()
    atype  = action.action_type
    target = (action.target or "").lower()
    diag   = action.parameters.get("diagnosis")
    done   = False

    # Diagnosis declaration (carried on any action)
    if diag and not ctx.diagnosis:
        ctx.diagnosis = diag
        if diag == DiagnosisTag.CRASH_LOOP:
            rb.diagnosis = 0.3
        else:
            rb.diagnosis = -0.3   # wrong hypothesis

    if atype in (ActionType.INSPECT_LOGS, ActionType.CHECK_METRICS):
        if target == "auth-service" and target not in ctx.inspected:
            rb.inspection = 0.2
        elif target and target not in ctx.inspected:
            rb.inspection = 0.05  # investigating a red herring
        ctx.inspected.add(target)
        return rb, done

    if atype == ActionType.ACKNOWLEDGE:
        rb.inspection = 0.05 if not ctx.diagnosis else 0.0
        return rb, done

    if atype == ActionType.RESTART_SERVICE:
        if target == "auth-service":
            rb.fix = 0.5
            done   = True
        elif target:
            rb.irrelevant = -0.1
        ctx.remediation_count += 1
        return rb, done

    if atype in (ActionType.ROLLBACK, ActionType.CLEAR_CACHE):
        rb.harmful = -0.3
        ctx.remediation_count += 1
        return rb, done

    if atype == ActionType.SCALE_UP:
        rb.irrelevant = -0.1
        ctx.remediation_count += 1
        return rb, done

    return rb, done


TASK_AUTH_CRASH = Task(
    id="auth-crash-loop",
    name="Auth Service Crash Loop",
    description=(
        "One or more services are reporting errors. "
        "Investigate logs and metrics, identify the root cause, "
        "and restore normal operation."
    ),
    difficulty="easy",
    max_steps=6,
    inject_fault=_inject_auth_crash,
    task_components=_components_auth_crash,
    true_root_cause=DiagnosisTag.CRASH_LOOP,
    true_fix_target="auth-service",
)


# ===========================================================================
# TASK 2 — MEDIUM  "payments-oom-cascade"
# ===========================================================================
# Root cause : payments-service OOM-killed (memory leak).
# Confusion  : api-gateway DEGRADED (downstream of payments — looks like the
#              root cause). db-proxy slow queries (red herring).
#              auth-service elevated CPU (noisy neighbour).
# Correct seq: INSPECT_LOGS payments-service (or check-metrics)
#              → INSPECT_LOGS / CHECK_METRICS api-gateway (optional, corroborating)
#              → RESTART_SERVICE payments-service (diagnosis=OOM_KILL)
# Partial fix: restarting api-gateway reduces visible errors briefly but
#              payments is still dead → partial reward, not done.
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
    rb     = RewardBreakdown()
    atype  = action.action_type
    target = (action.target or "").lower()
    diag   = action.parameters.get("diagnosis")
    done   = False

    if diag and not ctx.diagnosis:
        ctx.diagnosis = diag
        if diag == DiagnosisTag.OOM_KILL:
            rb.diagnosis = 0.3
        elif diag == DiagnosisTag.UPSTREAM_TIMEOUT:
            rb.diagnosis = -0.1    # partially reasonable — gateway IS timing out
        else:
            rb.diagnosis = -0.3

    if atype in (ActionType.INSPECT_LOGS, ActionType.CHECK_METRICS):
        if target == "payments-service" and target not in ctx.inspected:
            rb.inspection = 0.2    # key service
        elif target == "api-gateway" and target not in ctx.inspected:
            rb.inspection = 0.1    # downstream — corroborating
        elif target and target not in ctx.inspected:
            rb.inspection = 0.05
        ctx.inspected.add(target)
        return rb, done

    if atype == ActionType.ACKNOWLEDGE:
        rb.inspection = 0.03
        return rb, done

    if atype == ActionType.RESTART_SERVICE:
        if target == "payments-service":
            rb.fix = 0.5
            done   = True
        elif target == "api-gateway":
            # Partial fix: gateway restarts, looks better, but payments still down
            rb.partial_fix = 0.05
        else:
            rb.irrelevant = -0.1
        ctx.remediation_count += 1
        return rb, done

    if atype == ActionType.ROLLBACK:
        rb.harmful = -0.3
        ctx.remediation_count += 1
        return rb, done

    if atype in (ActionType.SCALE_UP, ActionType.CLEAR_CACHE):
        rb.irrelevant = -0.1
        ctx.remediation_count += 1
        return rb, done

    return rb, done


TASK_PAYMENTS_OOM = Task(
    id="payments-oom-cascade",
    name="Payments OOM Cascade",
    description=(
        "Multiple services are showing anomalies. "
        "Identify the true root cause, form a diagnosis, and apply the correct fix."
    ),
    difficulty="medium",
    max_steps=8,
    inject_fault=_inject_payments_oom,
    task_components=_components_payments_oom,
    true_root_cause=DiagnosisTag.OOM_KILL,
    true_fix_target="payments-service",
)


# ===========================================================================
# TASK 3 — HARD  "network-split-db-leak"
# ===========================================================================
#
# Scenario
# --------
# A network partition isolated db-proxy from its standby replica 4 hours ago.
# Since then, the primary has been running without replica acknowledgements,
# causing transaction log buildup (disk leak). Three symptoms now converge:
#
#   db-proxy      DEGRADED  — high disk I/O, growing WAL log, elevated latency
#   api-gateway   DEGRADED  — intermittent 503s when transactions hit write timeout
#   payments-svc  DEGRADED  — long-running DB writes timing out (not OOM this time)
#   user-service  DEGRADED  — read queries slow due to lock contention on db-proxy
#   auth-service  HEALTHY   — uses read replica shard, unaffected
#
# Three competing hypotheses the agent must rule out:
#
#   H1  OOM_KILL on payments-service
#       Evidence FOR: payments error rate high, memory not low this time (62%)
#       Evidence AGAINST: memory is not at limit; db-proxy is the common factor
#
#   H2  BAD_DEPLOY on api-gateway
#       Evidence FOR: api-gateway shows 503s, was deployed 6h ago
#       Evidence AGAINST: api-gateway logs show db write timeouts, not code errors
#
#   H3  RESOURCE_SATURATION on db-proxy  ← CORRECT
#       Evidence: WAL disk usage 94%, replica lag ∞, write latency 8.4s
#
# Partial fixes and delayed symptoms
# ------------------------------------
#   RESTART_SERVICE api-gateway   → gateway comes back for ~1 step, then degrades
#                                    again (db is still broken) — misleading
#   RESTART_SERVICE payments-svc  → same pattern: brief improvement, re-degrades
#   SCALE_UP db-proxy             → reduces CPU briefly, does NOT fix disk/WAL
#   ROLLBACK api-gateway          → makes things WORSE (adds a new bad deploy)
#
# Correct sequence (minimum 3 steps)
#   Step 1: INSPECT_LOGS db-proxy          → reveals WAL buildup + replica lag
#   Step 2: CHECK_METRICS db-proxy         → confirms disk 94%, write latency 8.4s
#           (or INSPECT_LOGS payments-svc to rule out OOM)
#   Step 3: CLEAR_CACHE db-proxy           → flushes WAL buffer, allows catchup
#           + parameters["diagnosis"] = RESOURCE_SATURATION
#
# Maximum score breakdown (per-step, before engine penalties):
#   inspect db-proxy (×1)            +0.20
#   corroborating inspect (×1)       +0.10
#   correct diagnosis                +0.30
#   correct fix after full inspect   +0.50
#   ─────────────────────────────────────
#   ceiling (before budget_bonus)    1.10  → clamped to 1.0
#
# ===========================================================================

def _inject_network_split(services: list[ServiceState]) -> list[ServiceState]:
    from .models import Metrics as M, ServiceState as S, ServiceStatus as SS

    # Add user-service (not in baseline 4)
    services.append(S(
        name="user-service", status=SS.DEGRADED,
        metrics=M(cpu_usage=44.0, memory_usage=51.0,
                  error_rate=9.0, latency_ms=3100.0, request_rate=88.0),
        version="2.3.1",
    ))

    for svc in services:
        if svc.name == "db-proxy":
            svc.status  = ServiceStatus.DEGRADED
            svc.metrics = Metrics(
                cpu_usage=82.0,
                memory_usage=78.0,
                error_rate=18.0,
                latency_ms=8400.0,   # write latency — 8.4 s
                request_rate=95.0,
            )
            svc.version = "14.5"     # postgres version — appears in logs

        elif svc.name == "api-gateway":
            svc.status  = ServiceStatus.DEGRADED
            svc.metrics = Metrics(
                cpu_usage=55.0, memory_usage=47.0,
                error_rate=28.0, latency_ms=6100.0, request_rate=195.0,
            )

        elif svc.name == "payments-service":
            svc.status  = ServiceStatus.DEGRADED
            svc.metrics = Metrics(
                cpu_usage=68.0,
                memory_usage=62.0,   # elevated but NOT at limit — not OOM
                error_rate=41.0,
                latency_ms=7800.0,
                request_rate=120.0,
            )

        elif svc.name == "auth-service":
            # Uses read-replica shard — unaffected by primary WAL issue
            # But emits noisy memory warning to confuse the agent
            svc.metrics.memory_usage = 81.0
            svc.metrics.latency_ms   = 145.0

    return services


# Signal logs exported for env.py log generation
NETWORK_SPLIT_SIGNALS: dict[str, list[str]] = {
    "db-proxy": [
        # True signals — WAL / replica lag
        "db-proxy: pg_wal directory size 47 GB — approaching disk limit (50 GB)",
        "db-proxy: standby replica lag: ∞ (replica unreachable since 04:12 UTC)",
        "db-proxy: WARNING checkpoint completion taking >30s — WAL write stall",
        "db-proxy: FATAL autovacuum worker killed — out of shared memory for WAL sender",
        "db-proxy: write latency P99 = 8412 ms (threshold: 500 ms)",
        # Conflicting signal — memory looks bad too (for H1 confusion)
        "db-proxy: shared_buffers usage 74% — consider increasing",
    ],
    "api-gateway": [
        # True signal — db write timeout, NOT a code bug
        "api-gateway: upstream db-proxy: write transaction timeout after 6000ms",
        "api-gateway: POST /v1/checkout → 503 Service Unavailable (db write failed)",
        "api-gateway: retry 2/2 exhausted for db write on route /v1/checkout",
        # Conflicting signal — makes BAD_DEPLOY look plausible
        "api-gateway: WARNING version mismatch detected in /health response headers",
        "api-gateway: last deployment: 6h ago (tag v3.1.4) — rollback available",
    ],
    "payments-service": [
        # True signal — db write timeout, NOT OOM
        "payments-service: DB write timeout after 7800ms for txn_id=TX991023",
        "payments-service: transaction commit stalled — waiting on db-proxy (8s)",
        # Conflicting signal — memory looks elevated (for H1 confusion)
        "payments-service: heap memory 62% — growth rate 0.3%/min (non-linear)",
        "payments-service: WARNING memory growth detected — monitoring",
    ],
    "user-service": [
        "user-service: SELECT latency 3100ms — lock contention on payments table",
        "user-service: read query queued 12s waiting for lock release from db-proxy",
    ],
    "auth-service": [
        # Red herring — auth is fine, just memory noise
        "auth-service: memory usage 81% — GC pressure from session cache growth",
        "auth-service: INFO read replica shard healthy — no issues detected",
    ],
}


def _components_network_split(
    action: Action,
    services: list[ServiceState],
    ctx: EpisodeContext,
) -> tuple[RewardBreakdown, bool]:
    rb     = RewardBreakdown()
    atype  = action.action_type
    target = (action.target or "").lower()
    diag   = action.parameters.get("diagnosis")
    done   = False

    # Diagnosis
    if diag and not ctx.diagnosis:
        ctx.diagnosis = diag
        if diag == DiagnosisTag.RESOURCE_SATURATION:
            rb.diagnosis = 0.3
        elif diag == DiagnosisTag.OOM_KILL:
            rb.diagnosis = -0.1      # plausible but wrong (memory elevated, not maxed)
        elif diag == DiagnosisTag.BAD_DEPLOY:
            rb.diagnosis = -0.1      # plausible but wrong (deployment logs mislead)
        else:
            rb.diagnosis = -0.3

    # Diagnostic actions
    if atype in (ActionType.INSPECT_LOGS, ActionType.CHECK_METRICS):
        if target == "db-proxy" and target not in ctx.inspected:
            rb.inspection = 0.2      # the key service
        elif target in ("api-gateway", "payments-service", "user-service") \
                and target not in ctx.inspected:
            rb.inspection = 0.1      # corroborating evidence, rules out H1/H2
        elif target == "auth-service" and target not in ctx.inspected:
            rb.inspection = 0.02     # red herring — nearly no signal
        elif target not in ctx.inspected:
            rb.inspection = 0.03
        ctx.inspected.add(target)
        return rb, done

    if atype == ActionType.ACKNOWLEDGE:
        rb.inspection = 0.02
        return rb, done

    # Remediation
    if atype == ActionType.CLEAR_CACHE:
        if target == "db-proxy":
            # Correct fix — reward scales with investigation depth
            db_inspected       = "db-proxy" in ctx.inspected
            corroborated       = bool(
                ctx.inspected & {"api-gateway", "payments-service", "user-service"}
            )
            if db_inspected and corroborated:
                rb.fix = 0.5         # full sequence
            elif db_inspected:
                rb.fix = 0.35        # partial investigation
            else:
                rb.fix = 0.15        # lucky blind clear
            done = True
        else:
            rb.irrelevant = -0.1
        ctx.remediation_count += 1
        return rb, done

    if atype == ActionType.RESTART_SERVICE:
        if target in ("api-gateway", "payments-service"):
            # Partial fix — service recovers briefly, then re-degrades next step
            # (The env applies this state change; the agent sees a false improvement)
            ctx.partial_fixes[target] = ctx.partial_fixes.get(target, 0) + 1
            rb.partial_fix = 0.05
        elif target == "db-proxy":
            # Restart doesn't flush WAL on disk — db comes back but re-saturates
            ctx.partial_fixes["db-proxy"] = ctx.partial_fixes.get("db-proxy", 0) + 1
            rb.partial_fix = 0.03
        elif target == "user-service":
            rb.partial_fix = 0.02
        elif target == "auth-service":
            rb.harmful = -0.3        # restarting a healthy service
        else:
            rb.irrelevant = -0.1
        ctx.remediation_count += 1
        return rb, done

    if atype == ActionType.SCALE_UP:
        if target == "db-proxy":
            # Reduces CPU briefly but not the root cause (WAL / disk)
            ctx.partial_fixes["scale_db"] = 1
            rb.partial_fix = 0.03
        else:
            rb.irrelevant = -0.1
        ctx.remediation_count += 1
        return rb, done

    if atype == ActionType.ROLLBACK:
        if target == "api-gateway":
            rb.harmful = -0.3        # rollback adds noise and reverts nothing useful
        else:
            rb.harmful = -0.3
        ctx.remediation_count += 1
        return rb, done

    return rb, done


TASK_NETWORK_SPLIT = Task(
    id="network-split-db-leak",
    name="Network Split — DB WAL Saturation",
    description=(
        "Multiple services are degraded. "
        "Three plausible root causes are present in the logs. "
        "Investigate carefully, eliminate false hypotheses, "
        "then apply the correct targeted fix."
    ),
    difficulty="hard",
    max_steps=10,
    inject_fault=_inject_network_split,
    task_components=_components_network_split,
    true_root_cause=DiagnosisTag.RESOURCE_SATURATION,
    true_fix_target="db-proxy",
)


# ===========================================================================
# Task loader
# ===========================================================================

TASK_REGISTRY: dict[str, Task] = {
    TASK_AUTH_CRASH.id:    TASK_AUTH_CRASH,
    TASK_PAYMENTS_OOM.id:  TASK_PAYMENTS_OOM,
    TASK_NETWORK_SPLIT.id: TASK_NETWORK_SPLIT,
}


def load_task(task_id: str) -> Task:
    if task_id not in TASK_REGISTRY:
        available = ", ".join(
            f"{t.id!r} ({t.difficulty})" for t in TASK_REGISTRY.values()
        )
        raise ValueError(f"Unknown task {task_id!r}. Available: {available}")
    return TASK_REGISTRY[task_id]


def list_tasks() -> list[dict]:
    return [
        {"id": t.id, "name": t.name, "difficulty": t.difficulty, "max_steps": t.max_steps}
        for t in TASK_REGISTRY.values()
    ]