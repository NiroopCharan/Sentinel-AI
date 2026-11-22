"""Deterministic process-instance and parent/child tracking.

Identity is scoped by host, boot, PID, process start time, and exec ID.
Events are processed in non-decreasing event-time order. Invalid transitions
are rejected before tracker state is mutated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .events import SyscallEvent


@dataclass(frozen=True)
class ProcessRecord:
    process_instance: str
    host_id: str
    boot_id: str
    pid: int
    ppid: int
    process_start: float
    exec_id: str
    first_seen: float
    last_seen: float
    parent_instance: Optional[str] = None
    terminated_at: Optional[float] = None
    termination_reason: Optional[str] = None


@dataclass(frozen=True)
class ProcessObservation:
    process: ProcessRecord
    parent_instance: Optional[str]
    pid_reuse_detected: bool


class ProcessTracker:
    """Track process lifecycle transitions in non-decreasing event-time order."""
    _EXIT_SYSCALLS = {"exit", "exit_group"}

    def __init__(self) -> None:
        self._records: dict[str, ProcessRecord] = {}
        self._active_by_pid: dict[tuple[str, str, int], str] = {}
        self._last_timestamp = -1.0

    @property
    def records(self) -> dict[str, ProcessRecord]:
        return dict(self._records)

    @property
    def active_processes(self) -> dict[tuple[str, str, int], str]:
        """Read-only copy of currently active PID slots."""
        return dict(self._active_by_pid)

    def observe(self, event: SyscallEvent) -> ProcessObservation:
        # Validate all rejection conditions before changing any internal state.
        if event.timestamp < 0 or event.process_start < 0:
            raise ValueError("Event and process start timestamps must be non-negative")
        if event.process_start > event.timestamp:
            raise ValueError("process_start cannot be later than event timestamp")
        if event.timestamp < self._last_timestamp:
            raise ValueError("Events must be observed in non-decreasing timestamp order")

        key = (event.host_id, event.boot_id, event.pid)
        identity = event.process_instance
        existing = self._records.get(identity)
        if existing is not None and existing.terminated_at is not None:
            raise ValueError(f"Observation received after process termination: {identity}")

        previous_id = self._active_by_pid.get(key)
        reused = previous_id is not None and previous_id != identity
        previous = self._records.get(previous_id) if reused else None

        parent_key = (event.host_id, event.boot_id, event.ppid)
        parent_id = (self._active_by_pid.get(parent_key)
                     if event.ppid != event.pid else None)

        if existing is None:
            record = ProcessRecord(
                process_instance=identity, host_id=event.host_id,
                boot_id=event.boot_id, pid=event.pid, ppid=event.ppid,
                process_start=event.process_start, exec_id=event.exec_id,
                first_seen=event.timestamp, last_seen=event.timestamp,
                parent_instance=parent_id,
            )
        else:
            record = ProcessRecord(**{**existing.__dict__, "last_seen": event.timestamp})

        syscall = event.syscall.lower()
        if syscall in self._EXIT_SYSCALLS:
            record = ProcessRecord(**{
                **record.__dict__, "terminated_at": event.timestamp,
                "termination_reason": syscall,
            })

        # Commit only after the proposed transition has been constructed.
        if previous is not None:
            self._records[previous_id] = ProcessRecord(**{
                **previous.__dict__, "terminated_at": event.timestamp,
                "termination_reason": "pid_reused",
            })
        if syscall in self._EXIT_SYSCALLS:
            self._active_by_pid.pop(key, None)
        else:
            self._active_by_pid[key] = identity
        self._records[identity] = record
        self._last_timestamp = event.timestamp
        return ProcessObservation(record, record.parent_instance, reused)

    def get(self, process_instance: str) -> Optional[ProcessRecord]:
        return self._records.get(process_instance)
