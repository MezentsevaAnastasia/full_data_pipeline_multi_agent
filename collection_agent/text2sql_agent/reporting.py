from __future__ import annotations

from collections import Counter
from pathlib import Path

from text2sql_agent.models import Text2SQLRecord


def write_report(
    output_path: Path,
    records: list[Text2SQLRecord],
    sources: list[str],
    dataset_path: Path,
    enriched: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_counts = Counter(record.source_name for record in records)
    dialect_counts = Counter(record.db_dialect for record in records)

    lines = [
        "# Text2SQL Dataset Report",
        "",
        "## Overview",
        "",
        f"- Records: {len(records)}",
        f"- Sources requested: {', '.join(sources)}",
        f"- Output dataset: `{dataset_path}`",
        f"- Enrichment enabled: {'yes' if enriched else 'no'}",
        "",
        "## Fields",
        "",
        "- `question`",
        "- `sql`",
        "- `schema`",
        "- `db_id`",
        "- `db_dialect`",
        "- `domain`",
        "- `difficulty`",
        "- `sql_features`",
        "- `source_url`",
        "- `source_name`",
        "- `collected_at`",
        "",
        "## Source Counts",
        "",
    ]

    for source_name, count in sorted(source_counts.items()):
        lines.append(f"- `{source_name}`: {count}")

    lines.extend(["", "## Dialects", ""])
    for dialect, count in sorted(dialect_counts.items()):
        lines.append(f"- `{dialect}`: {count}")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Duplicates are removed using `question + sql + db_id` when available.",
            "- `difficulty`, `domain`, and `sql_features` may be derived fields if enrichment is enabled.",
            "- `schema` is stored as a compact JSON string.",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
