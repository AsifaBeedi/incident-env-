"""
models.py — Pydantic data contracts for the AI Incident Response OpenEnv environment.
OpenEnv return signature
------------------------
    obs, reward, done, info = env.step(action)
    observation : Observation
    reward      : float              strictly in (0.0, 1.0)
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
    INSPECT_LOGS    = "inspect_logs"
    CHECK_METRICS   = "check_metrics"
    RESTART_SERVICE = "restart_service"
    SCALE_UP        = "scale_up"
    ROLLBACK        = "rollback"
    CLEAR_CACHE     = "clear_cache"
    ACKNOWLEDGE     = "acknowledge"
    NO_OP           = "no_op"


# Remediation action types — used by reward engine to enforce diagnosis-first
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
    logs:          list[LogEntry]
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


# ---------------------------------------------------------------------------
# RewardBreakdown
# ALL component defaults are 0.0.
# IMPORTANT: Only rb.final and rb.raw are guaranteed to be in (0.01, 0.99).
# Individual components may be 0.0 by design (they represent additive deltas).
# The validator must only check rb.final and rb.raw, not individual components.
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    # Task-logic components (additive deltas, may be 0.0)
    inspection:   float = 0.0
    diagnosis:    float = 0.0
    fix:          float = 0.0
    partial_fix:  float = 0.0
    harmful:      float = 0.0
    irrelevant:   float = 0.0
    # Engine penalty components (may be 0.0)
    repeat:       float = 0.0
    no_diagnosis: float = 0.0
    no_op:        float = 0.0
    # Engine bonus (may be 0.0)
    budget_bonus: float = 0.0
    # Totals — ALWAYS strictly in (0.01, 0.99), computed by reward_engine
    raw:          float = 0.01
    final:        float = 0.01
