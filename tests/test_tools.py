"""Tests for deterministic safety of tools and precondition validator."""

from __future__ import annotations

from pathlib import Path
import pytest

from forgeai.schemas import DocumentDigest, DocumentInfo, RequirementsDoc
from forgeai.tools.calculator import run_calculator, safe_calculate
from forgeai.tools.csv_lookup import run_csv_lookup
from forgeai.tools.current_date import run_current_date
from forgeai.tools.registry import check_tool_precondition


def test_calculator_safe_arithmetic_and_injection_blocks() -> None:
    assert safe_calculate("12 * 45.5 + 100") == 646.0
    assert safe_calculate("round(sqrt(16) + 2**3, 2)") == 12
    for malicious in (
        "__import__('os').system('echo pwn')",
        "open('/etc/passwd').read()",
        "lambda x: x",
        "2 ** 5000000",
        "().__class__.__mro__",
    ):
        with pytest.raises(ValueError):
            safe_calculate(malicious)
    out = run_calculator({"expression": "__import__('os')"})
    assert out["ok"] is False


def test_csv_lookup_filters_rows(tmp_path: Path) -> None:
    (tmp_path / "spares.csv").write_text(
        "part_number,description,unit_price_usd\nSP-0001,Impeller,150.0\nSP-0002,Gasket,25.0\n",
        encoding="utf-8",
    )
    res = run_csv_lookup({"column": "part_number", "value": "SP-0001"}, {"docs_dir": str(tmp_path)})
    assert res["ok"] is True
    assert res["total_matches"] == 1
    assert res["rows"][0]["description"] == "Impeller"


def test_current_date_returns_iso() -> None:
    out = run_current_date()
    assert out["ok"] is True
    assert len(out["date_iso"]) == 10


def test_tool_preconditions_block_unneeded_tools() -> None:
    req_hr = RequirementsDoc(
        agent_name="hr-policy-assistant",
        scope="Answer HR policy questions for shore staff",
        audience="Shore staff",
        tone="Polite",
        needs_numeric=False,
        needs_external_data=False,
        needs_tabular_lookup=False,
    )
    digest_no_csv = DocumentDigest(
        documents=[
            DocumentInfo(
                file_name="leave.pdf",
                file_type="pdf",
                size_bytes=100,
                num_chars=500,
                summary="Leave policy",
            )
        ],
        has_csv=False,
    )
    ok_csv, _ = check_tool_precondition("csv_lookup", req_hr, digest_no_csv)
    ok_calc, _ = check_tool_precondition("calculator", req_hr, digest_no_csv)
    ok_weather, _ = check_tool_precondition("http_weather", req_hr, digest_no_csv)
    assert not ok_csv
    assert not ok_calc
    assert not ok_weather
