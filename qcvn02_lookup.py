"""Lookup of seismic parameters transcribed from QCVN 02:2021/BXD tables 6.1/6.2."""
from __future__ import annotations

import json
from pathlib import Path

DATA_PATH = Path(__file__).with_name("qcvn02_2021_ag_lookup.json")
with DATA_PATH.open("r", encoding="utf-8") as stream:
    _RECORDS = json.load(stream)

_BY_PROVINCE: dict[str, list[dict]] = {}
_BY_KEY: dict[tuple[str, str], dict] = {}
for _record in _RECORDS:
    _BY_PROVINCE.setdefault(_record["province"], []).append(_record)
    _BY_KEY[(_record["province"], _record["admin"])] = _record


def provinces() -> list[str]:
    """Return province/city names as written in the 2021 regulation."""
    return sorted(_BY_PROVINCE)


def places(province: str) -> list[str]:
    """Return districts or equivalent administrative units for a province."""
    return sorted(row["admin"] for row in _BY_PROVINCE.get(province, []))


def lookup(province: str, admin: str) -> dict:
    """Return a copy of the QCVN row, including source page metadata."""
    row = _BY_KEY.get((province, admin))
    if row is None:
        raise KeyError(f"Không có địa danh {admin!r} thuộc {province!r} trong bảng 6.1.")
    return dict(row)

