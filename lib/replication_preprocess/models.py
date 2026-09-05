"""Exact-time helpers and compatibility exports for replication preprocessing."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from typing import Any

from .config import TOOL_IMPLEMENTATION_REVISION, load_config, validate_config
from .storage import sha256_json


def fraction_from_time_base(value: dict[str, Any] | Fraction) -> Fraction:
    if isinstance(value, Fraction):
        return value
    return Fraction(int(value["num"]), int(value["den"]))


def time_base_json(value: Fraction) -> dict[str, int]:
    return {"num": value.numerator, "den": value.denominator}


def seconds_fraction(pts: int, time_base: dict[str, Any] | Fraction) -> Fraction:
    return int(pts) * fraction_from_time_base(time_base)


def decimal_string(value: Fraction | Decimal, places: int = 9) -> str:
    if isinstance(value, Fraction):
        value = Decimal(value.numerator) / Decimal(value.denominator)
    quantized = value.quantize(Decimal(1).scaleb(-places))
    text = format(quantized, "f").rstrip("0").rstrip(".")
    return text or "0"


def time_point(pts: int, time_base: Fraction) -> dict[str, Any]:
    return {
        "pts": int(pts),
        "time_base": time_base_json(time_base),
        "seconds": decimal_string(seconds_fraction(pts, time_base)),
    }


def interval_duration(start_pts: int, end_pts: int, time_base: Fraction) -> Fraction:
    return (int(end_pts) - int(start_pts)) * time_base


def stable_id(prefix: str, *parts: Any, length: int = 16) -> str:
    return f"{prefix}_{sha256_json(list(parts))[:length]}"
