"""Structured numeric evidence extraction for matched legal clauses.

This module detects dates, amounts, percentages, and durations in matched master
and service clause text. It reports conservative structured evidence that can
augment an auditor model, but it does not decide whether a contradiction exists
and does not verify quotes; guardrail enforcement remains in ``auditor_guardrail``.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Literal

ValueType = Literal["date", "amount", "percentage", "duration"]
ComparisonStatus = Literal["match", "mismatch", "ambiguous"]

_MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}
_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "ninety": 90,
}
_CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP"}
_DURATION_UNITS = {
    "day": "day",
    "days": "day",
    "month": "month",
    "months": "month",
    "year": "year",
    "years": "year",
}
_AMOUNT_RE = re.compile(r"(?P<symbol>[$€£])\s*(?P<value>\d[\d,]*(?:\.\d+)?)")
_PERCENT_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?:%|percent|percentage points?)", re.IGNORECASE)
_NUMERIC_DATE_RE = re.compile(r"\b(?P<month>\d{1,2})/(?P<day>\d{1,2})/(?P<year>\d{2,4})\b")
_TEXT_DATE_RE = re.compile(
    r"\b(?P<month>January|Jan|February|Feb|March|Mar|April|Apr|May|June|Jun|July|Jul|August|Aug|"
    r"September|Sept|Sep|October|Oct|November|Nov|December|Dec)\s+"
    r"(?P<day>\d{1,2})(?:st|nd|rd|th)?[,]?\s+(?P<year>\d{4})\b",
    re.IGNORECASE,
)
_DURATION_RE = re.compile(
    r"\b(?P<value>\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|ninety)\s+"
    r"(?P<unit>days?|months?|years?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NumericValue:
    value_type: ValueType
    value: float | str
    unit: str
    text: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class NumericEvidence:
    value_type: ValueType
    status: ComparisonStatus
    master_value: NumericValue
    service_value: NumericValue
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "value_type": self.value_type,
            "status": self.status,
            "master_value": asdict(self.master_value),
            "service_value": asdict(self.service_value),
            "explanation": self.explanation,
        }


def extract_numeric_values(text: str) -> list[dict[str, Any]]:
    """Return typed numeric values found in ``text`` as JSON-serializable dicts."""
    return [asdict(value) for value in _extract_values(text)]


def compare_numeric_values(master_text: str, service_text: str) -> list[dict[str, Any]]:
    """Compare directly related numeric values across one master/service clause pair."""
    master_values = _extract_values(master_text)
    service_values = _extract_values(service_text)
    evidence: list[NumericEvidence] = []
    for value_type in ("date", "amount", "percentage", "duration"):
        master_by_type = [value for value in master_values if value.value_type == value_type]
        service_by_type = [value for value in service_values if value.value_type == value_type]
        if not master_by_type or not service_by_type:
            continue
        evidence.append(_compare_first(value_type, master_by_type[0], service_by_type[0]))
    return [item.to_dict() for item in evidence]


def evidence_for_retrieval_result(retrieval_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Build numeric evidence for each matched master chunk in a retrieval result."""
    service_text = str(retrieval_result.get("service_chunk_text", ""))
    evidence: list[dict[str, Any]] = []
    for master_chunk in retrieval_result.get("matched_master_chunks", []):
        master_text = str(master_chunk.get("master_chunk_text", ""))
        master_id = str(master_chunk.get("master_chunk_id", ""))
        for item in compare_numeric_values(master_text, service_text):
            evidence.append({"master_chunk_id": master_id, **item})
    return evidence


def _compare_first(value_type: ValueType, master: NumericValue, service: NumericValue) -> NumericEvidence:
    if master.unit != service.unit:
        return NumericEvidence(
            value_type,
            "ambiguous",
            master,
            service,
            f"Found {value_type} values with non-comparable units: {master.unit} vs {service.unit}.",
        )
    status: ComparisonStatus = "match" if master.value == service.value else "mismatch"
    return NumericEvidence(
        value_type,
        status,
        master,
        service,
        f"Found {value_type} {status}: master {master.text!r} vs service {service.text!r}.",
    )


def _extract_values(text: str) -> list[NumericValue]:
    values: list[NumericValue] = []
    values.extend(_extract_dates(text))
    values.extend(_extract_amounts(text))
    values.extend(_extract_percentages(text))
    values.extend(_extract_durations(text))
    values.sort(key=lambda item: (item.char_start, item.char_end, item.value_type))
    return values


def _extract_dates(text: str) -> list[NumericValue]:
    values: list[NumericValue] = []
    occupied: list[range] = []
    for match in _TEXT_DATE_RE.finditer(text):
        parsed = _safe_date(int(match.group("year")), _MONTHS[match.group("month").lower()], int(match.group("day")))
        if parsed is None:
            continue
        values.append(_value("date", parsed.isoformat(), "date", match))
        occupied.append(range(match.start(), match.end()))
    for match in _NUMERIC_DATE_RE.finditer(text):
        if any(match.start() in span or match.end() - 1 in span for span in occupied):
            continue
        year = int(match.group("year"))
        if year < 100:
            year += 2000
        parsed = _safe_date(year, int(match.group("month")), int(match.group("day")))
        if parsed is not None:
            values.append(_value("date", parsed.isoformat(), "date", match))
    return values


def _extract_amounts(text: str) -> list[NumericValue]:
    return [
        _value("amount", float(match.group("value").replace(",", "")), _CURRENCY_SYMBOLS[match.group("symbol")], match)
        for match in _AMOUNT_RE.finditer(text)
    ]


def _extract_percentages(text: str) -> list[NumericValue]:
    return [_value("percentage", float(match.group("value")), "percent", match) for match in _PERCENT_RE.finditer(text)]


def _extract_durations(text: str) -> list[NumericValue]:
    values: list[NumericValue] = []
    for match in _DURATION_RE.finditer(text):
        raw_value = match.group("value").lower()
        numeric = float(_NUMBER_WORDS.get(raw_value, raw_value))
        values.append(_value("duration", numeric, _DURATION_UNITS[match.group("unit").lower()], match))
    return values


def _value(value_type: ValueType, value: float | str, unit: str, match: re.Match[str]) -> NumericValue:
    return NumericValue(value_type, value, unit, match.group(0), match.start(), match.end())


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None
