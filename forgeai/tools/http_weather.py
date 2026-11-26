"""Keyless Open-Meteo weather forecast tool."""

from __future__ import annotations

from typing import Any

import httpx


def run_http_weather(
    args: dict[str, Any], _context: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Fetch current & daily forecast from Open-Meteo (keyless)."""
    lat = args.get("latitude", args.get("lat"))
    lon = args.get("longitude", args.get("lon"))
    location = args.get("location") or args.get("city")

    try:
        if (lat is None or lon is None) and location:
            geo_resp = httpx.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": str(location), "count": 1, "language": "en"},
                timeout=10.0,
            )
            geo_resp.raise_for_status()
            results = geo_resp.json().get("results") or []
            if not results:
                return {"ok": False, "error": f"Location not found: {location}"}
            lat = results[0]["latitude"]
            lon = results[0]["longitude"]
            location = f"{results[0].get('name')}, {results[0].get('country', '')}".strip(", ")

        if lat is None or lon is None:
            return {
                "ok": False,
                "error": "Provide latitude & longitude (or location name).",
            }

        resp = httpx.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": float(lat),
                "longitude": float(lon),
                "current": "temperature_2m,wind_speed_10m,precipitation",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "forecast_days": 3,
                "timezone": "auto",
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "ok": True,
            "location": location or f"{lat},{lon}",
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            "current": data.get("current", {}),
            "daily": data.get("daily", {}),
        }
    except Exception as exc:
        return {"ok": False, "error": f"Weather lookup failed: {exc}"}
