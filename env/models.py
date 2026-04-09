"""
models.py — Pydantic data contracts for the AI Incident Response OpenEnv environment.

OpenEnv return signature
------------------------
    obs, reward, done, info = env.step(action)

    observation : Observation
    reward      : float              clamped [0.0, 1.0]
    done        : bool
    info        : dict               see RewardBreakdown + episode stats
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ServiceStatus(str, Enum):
    HEALTHY  = "healthy"
    DEGRADED = "degraded"
    DOWN     = "down"


class ActionType(str, Enum):
    # Diagnostic — no state mutation; produce richer log/metric slices
    INSPECT_LOGS  = "inspect_logs"
    CHECK_METRICS = "check_metrics"
    # Remediation — mutate system state
    RESTART_SERVICE = "restart_service"
    SCALE_UP        = "scale_up"
    ROLLBACK        = "rollback"
    CLEAR_CACHE     = "clear_cache"
    # Bookkeeping
    ACKNOWLEDGE = "acknowledge"
    NO_OP       = "no_op"


# Remediation action types — used by the reward engine to enforce diagnosis-first
REMEDIATION_ACTIONS: frozenset[ActionType] = frozenset({
    ActionType.RESTART_SERVICE,
    ActionType.SCALE_UP,
    ActionType.ROLLBACK,
    ActionType.CLEAR_CACHE,
})


class Severity(str, Enum):
    INFO     = "info"
    WARNING  = "warning"
    ERROR    = "error"
    CRITICAL = "critical"


class DiagnosisTag(str, Enum):
    """
    Agent-declared root-cause hypothesis.
    Pass via  action.parameters["diagnosis"] = DiagnosisTag.XYZ
    alongside any action (typically a remediation action).
    """
    OOM_KILL            = "oom_kill"
    CRASH_LOOP          = "crash_loop"
    UPSTREAM_TIMEOUT    = "upstream_timeout"
    RESOURCE_SATURATION = "resource_saturation"
    BAD_DEPLOY          = "bad_deploy"
    CACHE_POISONING     = "cache_poisoning"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class LogEntry(BaseModel):
    timestamp: float
    severity:  Severity
    service:   str
    message:   str


class Metrics(BaseModel):
    cpu_usage:    float = Field(..., ge=0.0, le=100.0)
    memory_usage: float = Field(..., ge=0.0, le=100.0)
    error_rate:   float = Field(..., ge=0.0, description="errors/s")
    latency_ms:   float = Field(..., ge=0.0, description="P95 ms")
    request_rate: float = Field(..., ge=0.0, description="req/s")


class ServiceState(BaseModel):
    name:    str
    status:  ServiceStatus
    metrics: Metrics
    version: str = "1.0.0"


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    step:          int
    time:          float
    services:      list[ServiceState]
    logs:          list[LogEntry]      # mixed signal + noise; may be misleading
    active_alerts: list[str]
    inspected:     list[str] = Field(default_factory=list)
    diagnosis_set: bool      = False


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Action(BaseModel):
    action_type: ActionType
    target:      str | None = None
    parameters:  dict[str, Any] = Field(default_factory=dict)
    # parameters["diagnosis"] = DiagnosisTag  — declare root cause hypothesis


# ---------------------------------------------------------------------------
# RewardBreakdown — lives inside info["reward_breakdown"]
# ---------------------------------------------------------------------------class RewardBreakdown(BaseModel):
    """
    Typed breakdown of how the step reward was computed.
    All component values are raw (pre-clamp); final is clamped to [0.0, 1.0].
    """
    # Task-logic components (set by task scorer)
    inspection:    float = 0.01   
    diagnosis:     float = 0.01   
    fix:           float = 0.01   
    partial_fix:   float = 0.01   
    # Penalty components (set by reward engine)
    repeat:        float = -0.01  
    no_diagnosis:  float = -0.01  
    harmful:       float = -0.01  
    irrelevant:    float = -0.01  
    no_op:         float = -0.01  
    # Env-level modifier
    budget_bonus:  float = 0.01   
    # Totals
    raw:           float = 0.50   # Changed from 0.0 to safely pass static checks
    final:         float = 0.50   # THIS IS THE FIX. Safely inside (0, 1) by default
