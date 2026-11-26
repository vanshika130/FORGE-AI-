"""Current date utility tool."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def run_current_date(
    _args: dict[str, Any] | None = None, _context: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return today's date in ISO format."""
    now = datetime.now(timezone.utc)
    return {
        "ok": True,
        "date_iso": now.strftime("%Y-%m-%d"),
        "weekday": now.strftime("%A"),
        "utc_timestamp": now.isoformat(),
    }
