"""Data-quality rules shared by the Silver layer and the unit tests.

Keeping the rules as plain Python (and mirroring them in SQL in the Silver notebook)
means you can unit-test them in CI without a cluster.
"""
from __future__ import annotations

import re
from datetime import date, datetime

VALID_STATUSES = ("Received", "In Review", "Pending Info", "Approved", "Paid", "Denied")
_STATUS_LOOKUP = {s.lower().replace(" ", ""): s for s in VALID_STATUSES}
CLAIM_ID_RE = re.compile(r"^CLM-\d{6}$")
MEMBER_ID_RE = re.compile(r"^M\d{6}$")


def normalize_status(raw: str | None) -> str | None:
    """' paid ' -> 'Paid', 'pending info' -> 'Pending Info'. Unknown values -> None."""
    if raw is None:
        return None
    key = re.sub(r"\s+", "", raw).lower()
    return _STATUS_LOOKUP.get(key)


def parse_service_date(raw: str | None) -> date | None:
    """Accept ISO (2026-07-14) and US (07/14/2026) formats; anything else -> None."""
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    return None


def is_valid_claim_id(value: str | None) -> bool:
    return bool(value) and bool(CLAIM_ID_RE.match(value))


def is_valid_member_id(value: str | None) -> bool:
    return bool(value) and bool(MEMBER_ID_RE.match(value))
