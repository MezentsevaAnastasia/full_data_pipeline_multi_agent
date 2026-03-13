from __future__ import annotations

import json
import os
from typing import Iterable

from text2sql_agent.models import Text2SQLRecord

DEFAULT_MODEL = "gemini-2.0-flash"


def enrich_records(
    records: list[Text2SQLRecord],
    tasks: Iterable[str],
    model: str = DEFAULT_MODEL,
) -> list[Text2SQLRecord]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set")

    try:
        from google import genai
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'google-genai'. Install requirements before enrichment."
        ) from exc

    client = genai.Client(api_key=api_key)
    task_list = [task.strip() for task in tasks if task.strip()]

    for index, record in enumerate(records):
        context = _build_context(record)
        for task in task_list:
            result = _run_task(client, model, task, context)
            _apply_result(record, task, result)

        if (index + 1) % 25 == 0:
            print(f"Enriched {index + 1}/{len(records)} rows")

    return records


def _run_task(client, model: str, task: str, context: str):
    prompt = _build_prompt(task, context)
    response = client.models.generate_content(model=model, contents=prompt)
    return _parse_json_result(response.text)


def _build_context(record: Text2SQLRecord) -> str:
    return (
        f"QUESTION:\n{record.question}\n\n"
        f"SQL:\n{record.sql}\n\n"
        f"SCHEMA:\n{record.schema}\n"
    )


def _build_prompt(task: str, context: str) -> str:
    if task == "difficulty":
        instruction = (
            "Estimate SQL query difficulty as one of easy, medium, hard. "
            'Return strict JSON: {"result": "easy"}.'
        )
    elif task == "sql_features":
        instruction = (
            "Extract SQL feature tags from the query. Choose from join, group_by, "
            "order_by, aggregation, nested_query, having, limit, distinct, subquery, "
            "union, where. Return strict JSON: "
            '{"result": ["join", "aggregation"]}.'
        )
    elif task == "domain":
        instruction = (
            "Infer a short domain label such as education, finance, sports, sales, "
            'or healthcare. Return strict JSON: {"result": "finance"}.'
        )
    else:
        raise ValueError(f"Unsupported enrichment task: {task}")

    return f"{instruction}\n\n{context}"


def _parse_json_result(text: str):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    payload = json.loads(cleaned)
    return payload["result"]


def _apply_result(record: Text2SQLRecord, task: str, result) -> None:
    if task == "difficulty":
        record.difficulty = str(result)
    elif task == "domain":
        record.domain = str(result)
    elif task == "sql_features":
        record.sql_features = (
            json.dumps(result, ensure_ascii=False) if isinstance(result, list) else str(result)
        )
