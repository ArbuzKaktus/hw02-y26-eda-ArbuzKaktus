import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, BadZipFile

MAX_BYTES = 1024 * 1024 * 1024


def import_file(source: Path, target_dir: Path) -> Path:
    if not source.is_file():
        raise ValueError(f"Файл не найден: {source}")
    member_name = source.name
    if source.suffix.lower() == ".zip":
        with ZipFile(source) as archive:
            members = [
                info for info in archive.infolist()
                if not info.is_dir()
                and Path(info.filename).suffix.lower() in {".json", ".geojson"}
                and not info.filename.startswith("__MACOSX/")
            ]
            if len(members) != 1:
                raise ValueError("В ZIP должен быть ровно один JSON/GeoJSON. Выберите файл вручную.")
            info = members[0]
            if info.file_size > MAX_BYTES:
                raise ValueError("Файл больше 1 ГБ. Импортируйте выбранный срез самостоятельно.")
            payload = archive.read(info)
            member_name = Path(info.filename).name
    else:
        if source.suffix.lower() not in {".json", ".geojson"}:
            raise ValueError("Поддерживаются .json, .geojson и .zip.")
        if source.stat().st_size > MAX_BYTES:
            raise ValueError("Файл больше 1 ГБ. Импортируйте выбранный срез самостоятельно.")
        payload = source.read_bytes()
    data = json.loads(payload.decode("utf-8-sig"))
    records = data.get("features") if isinstance(data, dict) and data.get("type") == "FeatureCollection" else data
    if not isinstance(records, list) or not records or not all(isinstance(r, dict) for r in records):
        raise ValueError("Ожидается непустая GeoJSON FeatureCollection или список объектов.")
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / (Path(member_name).stem + ".geojson")
    digest = hashlib.sha256(payload).hexdigest()
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() == digest:
            return destination
        raise ValueError(f"Уже есть другой файл {destination.name}. Переименуйте новую выгрузку.")
    with destination.open("xb") as handle:
        handle.write(payload)
    metadata = {
        "imported_at_utc": datetime.now(timezone.utc).isoformat(),
        "downloaded_filename": source.name,
        "archive_member": member_name,
        "sha256": digest,
        "records": len(records),
        "source_page": "https://dtp-stat.ru/opendata/",
        "note": "Источник и дату скачивания подтвердите в solution/analysis.md.",
    }
    destination.with_suffix(".source.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Скачанный ZIP, GeoJSON или JSON")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "raw")
    args = parser.parse_args()
    try:
        destination = import_file(args.source.expanduser(), args.output_dir)
    except (OSError, ValueError, BadZipFile) as error:
        parser.exit(1, f"Не удалось импортировать: {error}\n")
    print(f"Данные готовы: {destination}")
    print("Запустите make run из корня репозитория.")


if __name__ == "__main__":
    main()
