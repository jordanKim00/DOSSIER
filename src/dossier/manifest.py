from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .common import read_jsonl, write_json


@dataclass
class ManifestItem:
    selected_index: int
    record_id: str
    set_id: int
    record_type: str
    level: int
    language: str
    question: str
    sample_id: str

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["set"] = self.set_id
        return payload


def load_records(path: Path) -> List[Dict[str, Any]]:
    return read_jsonl(path)


def build_manifest(records: Sequence[Dict[str, Any]]) -> List[ManifestItem]:
    manifest: List[ManifestItem] = []
    for index, record in enumerate(records):
        manifest.append(
            ManifestItem(
                selected_index=index,
                record_id=str(record.get("id", "unknown")),
                set_id=int(record.get("set", 0) or 0),
                record_type=str(record.get("type", "unknown")),
                level=int(record.get("level", 0) or 0),
                language=str(record.get("language", "unknown")),
                question=str(record.get("question", "")).strip(),
                sample_id=f"{record.get('type', 'unknown')}_level{record.get('level', 0)}_{index}",
            )
        )
    return manifest


def save_manifest(items: Sequence[ManifestItem], path: Path) -> Path:
    return write_json(path, [item.to_dict() for item in items])
