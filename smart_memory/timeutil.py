"""Parse the ``since`` / ``until`` arguments of search and context."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Optional, Union

TimeArg = Union[None, float, int, str, datetime, date, timedelta]

_UNITS = {"s": 1, "m": 60, "h": 3_600, "d": 86_400, "w": 604_800}
_RELATIVE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhdw])\s*$", re.IGNORECASE)


def to_timestamp(value: TimeArg, now: float) -> Optional[float]:
    """Epoch seconds for a time argument.

    Accepts epoch seconds, ``datetime`` / ``date``, ``timedelta`` (that long ago), relative
    strings such as ``"30m"``, ``"24h"``, ``"7d"``, ``"2w"`` (that long ago), or ISO dates
    (``"2026-11-01"``, ``"2026-11-01T09:30"``).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("expected a time, not a bool")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, timedelta):
        return now - value.total_seconds()
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day).timestamp()
    if isinstance(value, str):
        m = _RELATIVE.match(value)
        if m:
            return now - float(m.group(1)) * _UNITS[m.group(2).lower()]
        try:
            return datetime.fromisoformat(value.strip()).timestamp()
        except ValueError:
            raise ValueError(f"Can't read time {value!r}; use e.g. '7d', '24h' or '2026-11-01'") from None
    raise TypeError(f"Unsupported time value: {value!r}")
