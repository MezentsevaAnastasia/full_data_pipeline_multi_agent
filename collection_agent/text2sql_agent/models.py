from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Text2SQLRecord:
    question: str
    sql: str
    schema: str
    db_id: str = ""
    db_dialect: str = "unknown"
    domain: str = ""
    difficulty: str = ""
    sql_features: str = ""
    source_url: str = ""
    source_name: str = ""
    collected_at: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)
