"""Strict JSONL ingestion for normalized syscall-event records."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, TextIO

from .events import SyscallEvent


class EventParseError(ValueError):
    """A malformed or out-of-order event, annotated with source and line."""


def parse_event(row: object, *, source: str = "<memory>", line_number: int = 1) -> SyscallEvent:
    if not isinstance(row, dict):
        raise EventParseError(f"{source}:{line_number}: expected a JSON object")
    try:
        return SyscallEvent.from_dict(row)
    except (TypeError, ValueError, KeyError) as exc:
        raise EventParseError(f"{source}:{line_number}: {exc}") from exc


def _check_order(event: SyscallEvent, previous: float | None,
                 *, source: str, line_number: int) -> None:
    if previous is not None and event.timestamp < previous:
        raise EventParseError(
            f"{source}:{line_number}: timestamp {event.timestamp} is earlier "
            f"than previous timestamp {previous}"
        )


def iter_jsonl(stream: TextIO, *, source: str = "<stream>") -> Iterator[SyscallEvent]:
    """Yield validated events; blank/comment lines are ignored.

    Timestamps must be non-decreasing across non-comment records. Equal
    timestamps are allowed. The error points to the offending source line.
    """
    previous_timestamp: float | None = None
    for line_number, raw in enumerate(stream, start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise EventParseError(
                f"{source}:{line_number}: invalid JSON at column {exc.colno}: {exc.msg}"
            ) from exc
        event = parse_event(row, source=source, line_number=line_number)
        _check_order(event, previous_timestamp, source=source, line_number=line_number)
        previous_timestamp = event.timestamp
        yield event


def load_jsonl(path: str | Path) -> list[SyscallEvent]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as stream:
        return list(iter_jsonl(stream, source=str(file_path)))


def parse_rows(rows: Iterable[dict]) -> list[SyscallEvent]:
    parsed: list[SyscallEvent] = []
    previous_timestamp: float | None = None
    for line_number, row in enumerate(rows, start=1):
        event = parse_event(row, line_number=line_number)
        _check_order(event, previous_timestamp, source="<memory>", line_number=line_number)
        parsed.append(event)
        previous_timestamp = event.timestamp
    return parsed
