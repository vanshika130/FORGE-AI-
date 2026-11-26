"""Machine-readable tool registry and precondition validator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from forgeai.schemas import DocumentDigest, RequirementsDoc
from forgeai.tools.calculator import run_calculator
from forgeai.tools.csv_lookup import run_csv_lookup
from forgeai.tools.current_date import run_current_date
from forgeai.tools.http_weather import run_http_weather


@dataclass(frozen=True)
class ToolSpec:
    """Specification and executor for a ForgeAI agent tool."""

    name: str
    purpose: str
    attach_when: str
    args_schema: dict[str, Any]
    executor: Callable[[dict[str, Any], dict[str, Any] | None], dict[str, Any]]


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "csv_lookup": ToolSpec(
        name="csv_lookup",
        purpose="Filter or search rows in a CSV/XLSX table by column and value or query string.",
        attach_when="A CSV or XLSX file exists in docs AND the brief needs structured lookups.",
        args_schema={
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "Optional filename"},
                "column": {"type": "string", "description": "Column name to filter"},
                "value": {"type": "string", "description": "Value to match in column"},
                "query": {"type": "string", "description": "Free-text search across columns"},
                "limit": {"type": "integer", "default": 10},
            },
        },
        executor=run_csv_lookup,
    ),
    "calculator": ToolSpec(
        name="calculator",
        purpose="Evaluate safe arithmetic expressions (totals, costs, quantities, unit conversions).",
        attach_when="The brief involves totals, costs, quantities, rates, or numerical calculations.",
        args_schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression e.g. '12 * 45.50 + 100'",
                }
            },
            "required": ["expression"],
        },
        executor=run_calculator,
    ),
    "http_weather": ToolSpec(
        name="http_weather",
        purpose="Fetch live Open-Meteo weather forecast by latitude/longitude or location name.",
        attach_when="The brief explicitly needs live weather or location forecast conditions.",
        args_schema={
            "type": "object",
            "properties": {
                "latitude": {"type": "number"},
                "longitude": {"type": "number"},
                "location": {"type": "string"},
            },
        },
        executor=run_http_weather,
    ),
    "current_date": ToolSpec(
        name="current_date",
        purpose="Return today's date (ISO YYYY-MM-DD and weekday).",
        attach_when="The brief involves deadlines, validity periods, lead-time dates, or 'as of today' logic.",
        args_schema={"type": "object", "properties": {}},
        executor=run_current_date,
    ),
}


def format_registry_for_prompt() -> str:
    """Format all registered tools into a concise prompt description."""
    lines: list[str] = []
    for spec in TOOL_REGISTRY.values():
        lines.append(
            f"- {spec.name}: {spec.purpose}\n"
            f"  Attach ONLY when: {spec.attach_when}\n"
            f"  Args schema: {spec.args_schema}"
        )
    return "\n".join(lines)


def check_tool_precondition(
    tool_name: str, requirements: RequirementsDoc, digest: DocumentDigest
) -> tuple[bool, str]:
    """Deterministically verify whether a proposed tool's preconditions hold."""
    if tool_name not in TOOL_REGISTRY:
        return False, f"Unknown tool '{tool_name}' not in registry."

    if tool_name == "csv_lookup":
        if not digest.has_csv:
            return False, "No CSV/XLSX document exists in the doc folder."
        if not requirements.needs_tabular_lookup:
            return False, "Requirements flag needs_tabular_lookup is false."
        return True, "OK"

    if tool_name == "calculator":
        if not requirements.needs_numeric:
            return False, "Requirements flag needs_numeric is false."
        return True, "OK"

    if tool_name == "http_weather":
        text_blob = (
            requirements.scope
            + " "
            + " ".join(requirements.must_answer)
            + " "
            + " ".join(c.a for c in requirements.clarifications)
        ).lower()
        weather_terms = ("weather", "forecast", "temperature", "wind", "rain", "storm", "climate")
        if not requirements.needs_external_data or not any(w in text_blob for w in weather_terms):
            return False, "Brief does not explicitly request live weather/forecast data."
        return True, "OK"

    if tool_name == "current_date":
        text_blob = (
            requirements.scope
            + " "
            + " ".join(requirements.must_answer)
            + " "
            + " ".join(c.a for c in requirements.clarifications)
        ).lower()
        date_terms = (
            "today",
            "deadline",
            "valid",
            "validity",
            "lead time",
            "lead-time",
            "expir",
            "as of",
            "current date",
        )
        if not requirements.needs_current_date and not any(t in text_blob for t in date_terms):
            return False, "Brief does not require current date or deadline/validity logic."
        return True, "OK"

    return False, "Unhandled tool precondition."


def execute_tool(
    tool_name: str, args: dict[str, Any], context: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Execute a registered tool safely."""
    spec = TOOL_REGISTRY.get(tool_name)
    if spec is None:
        return {"ok": False, "error": f"Tool '{tool_name}' is not registered."}
    return spec.executor(args, context)
