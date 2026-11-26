"""CSV / XLSX tabular lookup and filtering tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    return pd.read_excel(path)


def run_csv_lookup(
    args: dict[str, Any], context: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Filter or search rows in CSV/XLSX files located in docs_dir."""
    ctx = context or {}
    docs_dir = Path(ctx.get("docs_dir", "."))
    if not docs_dir.exists():
        return {"ok": False, "error": f"docs_dir not found: {docs_dir}"}

    tables = sorted(
        p
        for p in docs_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".csv", ".xlsx", ".xls"}
    )
    if not tables:
        return {"ok": False, "error": "No CSV or XLSX files found in docs_dir"}

    requested_file = args.get("file")
    target = tables[0]
    if requested_file:
        matches = [p for p in tables if p.name == requested_file]
        if matches:
            target = matches[0]
        else:
            return {
                "ok": False,
                "error": f"File '{requested_file}' not found. Available: {[p.name for p in tables]}",
            }

    try:
        df = _load_table(target)
    except Exception as exc:
        return {"ok": False, "error": f"Failed to load {target.name}: {exc}"}

    column = args.get("column")
    value = args.get("value")
    query = args.get("query")
    limit = min(max(int(args.get("limit", 10)), 1), 25)

    filtered = df
    if column and value is not None:
        col_matches = [c for c in df.columns if str(c).lower() == str(column).lower()]
        if not col_matches:
            return {
                "ok": False,
                "error": f"Column '{column}' not found. Columns: {list(df.columns)}",
            }
        col_name = col_matches[0]
        series_str = df[col_name].astype(str).str.lower()
        val_str = str(value).strip().lower()
        exact = df[series_str == val_str]
        filtered = exact if not exact.empty else df[series_str.str.contains(val_str, na=False, regex=False)]
    elif query:
        q_str = str(query).strip().lower()
        mask = pd.Series(False, index=df.index)
        for col in df.columns:
            mask = mask | df[col].astype(str).str.lower().str.contains(q_str, na=False, regex=False)
        filtered = df[mask]

    rows = filtered.head(limit).fillna("").to_dict(orient="records")
    return {
        "ok": True,
        "file": target.name,
        "columns": [str(c) for c in df.columns],
        "total_matches": int(len(filtered)),
        "rows": rows,
    }
