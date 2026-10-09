import csv
import pathlib
from datetime import date

import pytest

from medinova.quality import is_valid_claim_id, is_valid_member_id, normalize_status, parse_service_date

CLAIMS = pathlib.Path(__file__).resolve().parent.parent / "sample_data" / "claims"


@pytest.mark.parametrize("raw,expected", [
    ("Paid", "Paid"), (" paid ", "Paid"), ("pending info", "Pending Info"),
    ("IN REVIEW", "In Review"), ("unknown", None), (None, None),
])
def test_normalize_status(raw, expected):
    assert normalize_status(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("2026-07-14", date(2026, 7, 14)), ("07/14/2026", date(2026, 7, 14)),
    ("14.07.2026", None), ("", None), (None, None),
])
def test_parse_service_date(raw, expected):
    assert parse_service_date(raw) == expected


def test_id_validators():
    assert is_valid_claim_id("CLM-100042")
    assert not is_valid_claim_id("CLM-42")
    assert is_valid_member_id("M100007")
    assert not is_valid_member_id("100007")


def test_sample_claims_are_cleanable():
    """Every row in the sample batches must survive the Silver rules."""
    for f in sorted(CLAIMS.glob("claims_batch_*.csv")):
        for row in csv.DictReader(f.open(encoding="utf-8")):
            assert is_valid_claim_id(row["claim_id"]), row
            assert is_valid_member_id(row["member_id"]), row
            assert normalize_status(row["claim_status"]) is not None, row
            assert parse_service_date(row["service_date"]) is not None, row
