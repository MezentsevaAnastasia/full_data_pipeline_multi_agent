from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from text2sql_agent.models import Text2SQLRecord

SOURCE_CANDIDATES = {
    "spider": ["spider", "xlangai/spider"],
    "wikisql": ["wikisql", "Salesforce/wikisql"],
}

SPLIT_CANDIDATES = ["train", "validation", "dev", "test"]


def load_source_records(
    source_name: str,
    split: str | None = None,
    limit: int | None = None,
    domain_filter: str | None = None,
) -> list[Text2SQLRecord]:
    if source_name not in SOURCE_CANDIDATES:
        raise ValueError(
            f"Unsupported source '{source_name}'. Supported: {', '.join(SOURCE_CANDIDATES)}"
        )

    dataset_rows, dataset_id, used_split = _load_dataset_rows(
        SOURCE_CANDIDATES[source_name], split
    )

    raw_records = _normalize_rows(source_name, dataset_rows, dataset_id)
    filtered_records = _filter_records(raw_records, domain_filter)

    if limit is not None:
        filtered_records = filtered_records[:limit]

    if not filtered_records:
        raise ValueError(
            f"No records extracted from source '{source_name}' using split '{used_split}'"
        )

    return filtered_records


def _load_dataset_rows(
    dataset_candidates: list[str], requested_split: str | None
) -> tuple[list[dict[str, Any]], str, str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'datasets'. Install requirements before running the agent."
        ) from exc

    split_candidates = [requested_split] if requested_split else SPLIT_CANDIDATES

    last_error: Exception | None = None
    for dataset_id in dataset_candidates:
        for split_name in split_candidates:
            try:
                dataset = load_dataset(dataset_id, split=split_name)
                return list(dataset), dataset_id, split_name
            except Exception as exc:  # pragma: no cover - network/runtime dependent
                last_error = exc

    raise RuntimeError(
        f"Failed to load any dataset candidate: {dataset_candidates}. Last error: {last_error}"
    )


def _normalize_rows(
    source_name: str, rows: list[dict[str, Any]], dataset_id: str
) -> list[Text2SQLRecord]:
    if source_name == "spider":
        return [_normalize_spider_row(row, dataset_id) for row in rows]
    if source_name == "wikisql":
        return [_normalize_wikisql_row(row, dataset_id) for row in rows]

    raise ValueError(f"Unsupported source: {source_name}")


def _normalize_spider_row(row: dict[str, Any], dataset_id: str) -> Text2SQLRecord:
    schema_payload = {
        "db_id": row.get("db_id", ""),
        "tables": row.get("db_table_names", []),
        "columns": row.get("db_column_names", []),
        "column_types": row.get("db_column_types", []),
        "primary_keys": row.get("db_primary_keys", []),
        "foreign_keys": row.get("db_foreign_keys", []),
    }

    return Text2SQLRecord(
        question=str(row.get("question", "")).strip(),
        sql=str(row.get("query", "")).strip(),
        schema=json.dumps(schema_payload, ensure_ascii=False),
        db_id=str(row.get("db_id", "")).strip(),
        db_dialect="sqlite",
        domain="general",
        source_url=f"https://huggingface.co/datasets/{dataset_id}",
        source_name="spider",
        collected_at=_utc_now(),
    )


def _normalize_wikisql_row(row: dict[str, Any], dataset_id: str) -> Text2SQLRecord:
    table = row.get("table", {}) or {}
    sql_struct = row.get("sql", {}) or {}

    schema_payload = {
        "table_name": table.get("name", ""),
        "header": table.get("header", []),
        "types": table.get("types", []),
    }

    return Text2SQLRecord(
        question=str(row.get("question", "")).strip(),
        sql=_wikisql_to_sql(sql_struct, table),
        schema=json.dumps(schema_payload, ensure_ascii=False),
        db_id=str(table.get("id", row.get("table_id", ""))).strip(),
        db_dialect="sqlite",
        domain="general",
        source_url=f"https://huggingface.co/datasets/{dataset_id}",
        source_name="wikisql",
        collected_at=_utc_now(),
    )


def _wikisql_to_sql(sql_struct: dict[str, Any], table: dict[str, Any]) -> str:
    headers = table.get("header", []) or []
    table_name = table.get("name") or table.get("id") or "table"

    agg_map = {
        0: "",
        1: "MAX",
        2: "MIN",
        3: "COUNT",
        4: "SUM",
        5: "AVG",
    }
    op_map = {
        0: "=",
        1: ">",
        2: "<",
        3: "OP",
    }

    select_idx = sql_struct.get("sel", 0)
    select_column = _safe_header(headers, select_idx)
    aggregation = agg_map.get(sql_struct.get("agg", 0), "")

    if aggregation:
        select_expression = f"{aggregation}({select_column})"
    else:
        select_expression = select_column

    conditions = []
    for condition in sql_struct.get("conds", []) or []:
        if len(condition) != 3:
            continue
        column_idx, operator_idx, value = condition
        operator = op_map.get(operator_idx, "=")
        column_name = _safe_header(headers, column_idx)
        conditions.append(f"{column_name} {operator} {_quote_sql_value(value)}")

    where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    return f"SELECT {select_expression} FROM {table_name}{where_clause}"


def _safe_header(headers: list[Any], index: int) -> str:
    try:
        return str(headers[index]).strip().replace(" ", "_")
    except Exception:
        return f"column_{index}"


def _quote_sql_value(value: Any) -> str:
    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def _filter_records(
    records: list[Text2SQLRecord], domain_filter: str | None
) -> list[Text2SQLRecord]:
    valid_records = [record for record in records if record.question and record.sql]
    if not domain_filter:
        return valid_records

    needle = domain_filter.lower()
    filtered = []
    for record in valid_records:
        haystack = " ".join([record.question, record.schema, record.domain]).lower()
        if needle in haystack:
            filtered.append(record)
    return filtered


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
