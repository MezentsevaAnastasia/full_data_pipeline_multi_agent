from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from text2sql_agent.enrichment import DEFAULT_MODEL, enrich_records
from text2sql_agent.models import Text2SQLRecord
from text2sql_agent.reporting import write_report
from text2sql_agent.sources import load_source_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect and normalize Text2SQL benchmark data into a single dataset."
    )
    parser.add_argument(
        "--sources",
        default="spider,wikisql",
        help="Comma-separated source names. Supported: spider,wikisql",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split to use. If omitted, the agent tries common splits automatically.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=200,
        help="Maximum number of final records.",
    )
    parser.add_argument(
        "--domain-filter",
        default=None,
        help="Optional keyword filter applied to question/schema text.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory for raw files, final dataset, and report.",
    )
    parser.add_argument(
        "--dataset-name",
        default="text2sql_dataset.csv",
        help="Final dataset file name.",
    )
    parser.add_argument(
        "--enrich",
        action="store_true",
        help="Use Gemini to derive extra fields such as difficulty and sql_features.",
    )
    parser.add_argument(
        "--enrich-tasks",
        default="difficulty,sql_features",
        help="Comma-separated enrichment tasks: difficulty,sql_features,domain",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="LLM model for enrichment.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    workspace_cache = Path(".hf_cache")
    os.environ.setdefault("HF_HOME", str(workspace_cache))
    os.environ.setdefault("HF_HUB_CACHE", str(workspace_cache / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(workspace_cache / "datasets"))

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    source_names = [item.strip() for item in args.sources.split(",") if item.strip()]
    per_source_limit = max(1, args.max_records // max(1, len(source_names)))

    all_records: list[Text2SQLRecord] = []
    for source_name in source_names:
        print(f"Loading source: {source_name}")
        records = load_source_records(
            source_name=source_name,
            split=args.split,
            limit=per_source_limit,
            domain_filter=args.domain_filter,
        )
        _write_raw_jsonl(raw_dir / f"{source_name}.jsonl", records)
        all_records.extend(records)
        print(f"Collected {len(records)} rows from {source_name}")

    unique_records = _deduplicate_records(all_records)[: args.max_records]

    if args.enrich:
        task_list = [item.strip() for item in args.enrich_tasks.split(",") if item.strip()]
        print(f"Running enrichment tasks: {', '.join(task_list)}")
        unique_records = enrich_records(unique_records, task_list, model=args.model)

    dataset_path = output_dir / args.dataset_name
    _write_dataset(dataset_path, unique_records)

    report_path = output_dir / "README.md"
    write_report(report_path, unique_records, source_names, dataset_path, args.enrich)

    print(f"Saved dataset: {dataset_path}")
    print(f"Saved report: {report_path}")
    print(f"Final rows: {len(unique_records)}")
    return 0


def _write_raw_jsonl(path: Path, records: list[Text2SQLRecord]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def _deduplicate_records(records: list[Text2SQLRecord]) -> list[Text2SQLRecord]:
    seen: set[tuple[str, str, str]] = set()
    unique = []

    for record in records:
        key = (record.question.strip(), record.sql.strip(), record.db_id.strip())
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)

    return unique


def _write_dataset(path: Path, records: list[Text2SQLRecord]) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'pandas'. Install requirements before running the agent."
        ) from exc

    frame = pd.DataFrame([record.to_dict() for record in records])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


if __name__ == "__main__":
    raise SystemExit(main())
