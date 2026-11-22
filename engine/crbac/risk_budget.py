"""Risk-budgeted CRBAC response policy. Decisions only; never executes OS actions."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite


@dataclass(frozen=True)
class RiskBudgetConfig:
    capacity: float = 1.0
    initial_budget: float = 1.0
    refill_per_second: float = 0.02
    anomaly_weight: float = 0.50
    causal_weight: float = 0.30
    persistence_weight: float = 0.20
    alert_threshold: float = 0.35
    quarantine_threshold: float = 0.55
    terminate_threshold: float = 0.80
    alert_cost: float = 0.05
    quarantine_cost: float = 0.30
    terminate_cost: float = 0.70

    def __post_init__(self):
        values = (
            self.capacity, self.initial_budget, self.refill_per_second,
            self.anomaly_weight, self.causal_weight, self.persistence_weight,
            self.alert_threshold, self.quarantine_threshold,
            self.terminate_threshold, self.alert_cost,
            self.quarantine_cost, self.terminate_cost,
        )
        if not all(isfinite(x) for x in values):
            raise ValueError("Configuration values must be finite")
        if self.capacity <= 0 or not 0 <= self.initial_budget <= self.capacity:
            raise ValueError("Invalid budget capacity or initial budget")
        if self.refill_per_second < 0:
            raise ValueError("Refill rate cannot be negative")
        if min(self.anomaly_weight, self.causal_weight,
               self.persistence_weight) < 0:
            raise ValueError("Risk weights cannot be negative")
        if abs(self.anomaly_weight + self.causal_weight +
               self.persistence_weight - 1.0) > 1e-9:
            raise ValueError("Risk weights must sum to 1")
        if not (0 <= self.alert_threshold <= self.quarantine_threshold
                <= self.terminate_threshold <= 1):
            raise ValueError("Invalid risk thresholds")
        if min(self.alert_cost, self.quarantine_cost,
               self.terminate_cost) < 0:
            raise ValueError("Action costs cannot be negative")


@dataclass(frozen=True)
class RiskDecision:
    timestamp: float
    process_instance: str
    risk_score: float
    requested_action: str
    selected_action: str
    budget_before: float
    budget_after: float
    budget_limited: bool
    protected_process: bool
    dry_run: bool = True


@dataclass
class RiskBudgetEngine:
    config: RiskBudgetConfig = field(default_factory=RiskBudgetConfig)
    protected_processes: frozenset[str] = frozenset()
    budget: float = field(init=False)
    last_timestamp: float = field(default=-1.0, init=False)

    def __post_init__(self):
        self.budget = self.config.initial_budget

    @staticmethod
    def _validate_score(name: str, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        value = float(value)
        if not isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and between 0 and 1")
        return value

    def _requested_action(self, risk: float) -> str:
        c = self.config
        if risk >= c.terminate_threshold:
            return "terminate"
        if risk >= c.quarantine_threshold:
            return "quarantine"
        if risk >= c.alert_threshold:
            return "alert"
        return "monitor"

    def _cost(self, action: str) -> float:
        return {
            "monitor": 0.0,
            "alert": self.config.alert_cost,
            "quarantine": self.config.quarantine_cost,
            "terminate": self.config.terminate_cost,
        }[action]

    def evaluate(
        self,
        timestamp: float,
        process_instance: str,
        anomaly_score: float,
        causal_score: float = 0.0,
        persistence_score: float = 0.0,
    ) -> RiskDecision:
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
            raise TypeError("timestamp must be numeric")
        timestamp = float(timestamp)
        if not isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp must be finite and non-negative")
        if timestamp < self.last_timestamp:
            raise ValueError("Events must be evaluated in timestamp order")
        if not isinstance(process_instance, str) or not process_instance.strip():
            raise ValueError("process_instance must be non-empty")

        anomaly = self._validate_score("anomaly_score", anomaly_score)
        causal = self._validate_score("causal_score", causal_score)
        persistence = self._validate_score("persistence_score", persistence_score)

        c = self.config
        elapsed = 0.0 if self.last_timestamp < 0 else timestamp - self.last_timestamp
        self.budget = min(c.capacity, self.budget + elapsed * c.refill_per_second)
        before = self.budget
        self.last_timestamp = timestamp

        risk = (
            c.anomaly_weight * anomaly
            + c.causal_weight * causal
            + c.persistence_weight * persistence
        )
        requested = self._requested_action(risk)
        protected = process_instance in self.protected_processes

        if protected and requested in ("quarantine", "terminate"):
            selected = "alert"
        else:
            selected = requested

        # Downgrade to the strongest action that fits the available budget.
        order = {
            "terminate": ("terminate", "quarantine", "alert", "monitor"),
            "quarantine": ("quarantine", "alert", "monitor"),
            "alert": ("alert", "monitor"),
            "monitor": ("monitor",),
        }
        budget_limited = False
        for candidate in order[selected]:
            if self._cost(candidate) <= self.budget + 1e-12:
                selected = candidate
                break
            budget_limited = True

        self.budget = max(0.0, self.budget - self._cost(selected))
        return RiskDecision(
            timestamp=timestamp,
            process_instance=process_instance,
            risk_score=risk,
            requested_action=requested,
            selected_action=selected,
            budget_before=before,
            budget_after=self.budget,
            budget_limited=budget_limited,
            protected_process=protected,
            dry_run=True,
        )
