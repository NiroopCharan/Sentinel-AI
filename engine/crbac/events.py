"""Validated normalized syscall-event schema and stable process identity."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Any


@dataclass(frozen=True)
class SyscallEvent:
    timestamp: float
    host_id: str
    boot_id: str
    pid: int
    process_start: float
    exec_id: str
    ppid: int
    syscall: str
    resource: str | None = None
    operation: str | None = None
    result: int | None = None

    @property
    def process_instance(self) -> str:
        return (f"{self.host_id}:{self.boot_id}:{self.pid}:"
                f"{self.process_start}:{self.exec_id}")

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "SyscallEvent":
        if not isinstance(row, dict):
            raise TypeError("event must be a dictionary")
        required = ("timestamp", "host_id", "boot_id", "pid", "process_start",
                    "exec_id", "ppid", "syscall")
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"Missing required event fields: {missing}")
        unknown = sorted(set(row) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown event fields: {unknown}")

        for key in ("timestamp", "process_start"):
            value = row[key]
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not isfinite(float(value))):
                raise ValueError(f"{key} must be a finite number")
            if value < 0:
                raise ValueError(f"{key} must be non-negative")
        if row["process_start"] > row["timestamp"]:
            raise ValueError("process_start cannot be later than event timestamp")

        for key in ("pid", "ppid"):
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{key} must be an integer")
            if value < 0:
                raise ValueError(f"{key} must be non-negative")
        for key in ("host_id", "boot_id", "exec_id", "syscall"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"{key} must be a non-empty string")
        for key in ("resource", "operation"):
            if row.get(key) is not None and not isinstance(row[key], str):
                raise ValueError(f"{key} must be a string or null")
        if row.get("result") is not None:
            value = row["result"]
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError("result must be an integer or null")

        return cls(
            timestamp=float(row["timestamp"]), host_id=row["host_id"],
            boot_id=row["boot_id"], pid=int(row["pid"]),
            process_start=float(row["process_start"]), exec_id=row["exec_id"],
            ppid=int(row["ppid"]), syscall=row["syscall"],
            resource=row.get("resource"), operation=row.get("operation"),
            result=row.get("result"),
        )
