import json
from pathlib import Path

import pandas as pd


def load_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict) and payload.get("type") == "FeatureCollection":
        records = payload.get("features")
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError("Ожидается GeoJSON FeatureCollection или список записей.")
    if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
        raise ValueError("Выгрузка должна содержать список объектов JSON.")
    return records


def properties(record: dict) -> dict:
    if record.get("type") == "Feature":
        value = record.get("properties")
        return value if isinstance(value, dict) else {}
    return record


def records_to_frame(records: list[dict]) -> pd.DataFrame:
    rows = []
    for index, record in enumerate(records):
        props = properties(record)
        row = {
            key: props.get(key)
            for key in (
                "id", "datetime", "region", "category", "severity", "light",
                "participants_count", "injured_count", "dead_count",
            )
        }
        row["source_row"] = index
        geometry = record.get("geometry")
        geometry = geometry if isinstance(geometry, dict) else {}
        coords = geometry.get("coordinates")
        if geometry.get("type") == "Point" and isinstance(coords, list) and len(coords) >= 2:
            row["longitude"], row["latitude"] = coords[:2]
        else:
            point = props.get("point")
            point = point if isinstance(point, dict) else {}
            row["longitude"], row["latitude"] = point.get("long"), point.get("lat")
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["datetime_raw"] = frame["datetime"]
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce", format="mixed")
    for name in ("longitude", "latitude"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame["map_valid"] = (
        frame["longitude"].between(-180, 180)
        & frame["latitude"].between(-90, 90)
        & ~((frame["longitude"] == 0) & (frame["latitude"] == 0))
    )
    return frame
