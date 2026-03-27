"""DataCollectionAgent — collects data from multiple sources into a unified DataFrame.

Supported source types:
  - hf_dataset  : HuggingFace Hub datasets
  - scrape      : Web scraping via CSS selectors (requests + BeautifulSoup)
  - api         : JSON REST APIs
"""

from __future__ import annotations

import os
import re
import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datasets import load_dataset as hf_load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lightweight keyword-based sentiment (used when auto_label=True)
# ---------------------------------------------------------------------------
_POS_WORDS = frozenset(
    "love good great happy beautiful wonderful best hope joy amazing excellent "
    "positive bright kind success inspire inspiring dream smile friend heart "
    "laugh light peace warm brave true truth wisdom freedom power life lovely "
    "pleasure delight cheerful gentle".split()
)
_NEG_WORDS = frozenset(
    "hate bad worst sad ugly terrible fear angry pain dark evil death fail wrong "
    "hurt war cry alone suffer hell misery enemy stupid lie despair broken lost "
    "weak nothing never regret trouble foolish danger cruel".split()
)


def _simple_sentiment(text: str) -> str:
    """Return 'positive' or 'negative' based on keyword overlap."""
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    return "positive" if len(words & _POS_WORDS) >= len(words & _NEG_WORDS) else "negative"


# ===========================================================================
# Agent
# ===========================================================================
class DataCollectionAgent:
    """Agent that collects data from heterogeneous sources and returns a
    unified :class:`pandas.DataFrame` with a fixed schema.

    Parameters
    ----------
    config : str | dict
        Path to a YAML config file **or** a config dictionary.
    """

    def __init__(self, config: str | dict = "config.yaml"):
        if isinstance(config, (str, Path)):
            with open(config, encoding="utf-8") as fh:
                self.config = yaml.safe_load(fh)
        else:
            self.config = config

        schema = self.config.get("schema", {})
        self.columns: list[str] = schema.get(
            "columns", ["text", "label", "source", "collected_at"]
        )

    # ── task discovery ─────────────────────────────────────────────────

    def available_tasks(self) -> dict[str, str]:
        """Return ``{task_name: description}`` for every task in config."""
        tasks = self.config.get("tasks", {})
        return {name: t.get("description", "") for name, t in tasks.items()}

    def _resolve_task(self, task: str) -> dict:
        """Look up a task by name and return its config block."""
        tasks = self.config.get("tasks", {})
        if task not in tasks:
            avail = ", ".join(tasks) or "(none)"
            raise KeyError(
                f"Task '{task}' not found in config.  Available: {avail}"
            )
        return tasks[task]

    # ── auto-discovery by topic ───────────────────────────────────────

    def discover_sources(
        self,
        topic: str,
        max_samples: int = 3000,
        max_candidates: int = 15,
    ) -> tuple[list[dict], dict]:
        """Search HuggingFace Hub for datasets matching a free-text *topic*.

        Automatically inspects dataset features to find the text column,
        label column, and label names.

        Returns
        -------
        sources : list[dict]
            Ready-to-use source descriptors for :meth:`run`.
        output_cfg : dict
            ``{"path": "data/raw/…csv", "format": "csv"}``.
        """
        from huggingface_hub import HfApi
        import datasets as ds_lib

        cache_dir = os.environ.get(
            "HF_HOME", os.path.join(os.getcwd(), ".hf_cache")
        )
        api = HfApi()

        words = topic.split()
        queries = [topic]
        if len(words) > 2:
            queries.append(" ".join(words[:3]))
            queries.append(" ".join(words[:2]))
        if len(words) > 1:
            queries.append(words[0])

        candidates = []
        used_query = topic
        for q in queries:
            logger.info("Searching HuggingFace Hub for '%s' …", q)
            candidates = list(
                api.list_datasets(search=q, sort="downloads", limit=max_candidates)
            )
            if candidates:
                used_query = q
                break

        if not candidates:
            raise RuntimeError(
                f"HuggingFace Hub: ничего не найдено по запросу '{topic}'"
            )
        logger.info(
            "  query='%s' → candidates: %s",
            used_query,
            ", ".join(c.id for c in candidates[:5]),
        )

        _TEXT_HINTS = (
            "text", "comment_text", "sentence", "content", "review",
            "question", "title", "body", "message", "tweet", "document",
        )
        _LABEL_HINTS = (
            "label", "labels", "target", "class", "sentiment", "category",
            "toxic", "rating", "score", "intent",
        )

        sources: list[dict] = []

        for cand in candidates:
            try:
                builder = ds_lib.load_dataset_builder(cand.id, cache_dir=cache_dir)
                features = builder.info.features
                if features is None:
                    continue

                text_field: str | None = None
                label_field: str | None = None
                label_map: dict | None = None

                str_cols = []
                classlabel_cols = []
                int_cols = []

                for fname, ftype in features.items():
                    if isinstance(ftype, ds_lib.ClassLabel):
                        classlabel_cols.append(fname)
                    elif isinstance(ftype, ds_lib.Value):
                        if ftype.dtype == "string":
                            str_cols.append(fname)
                        elif ftype.dtype in ("int32", "int64", "int8", "int16", "float32", "float64"):
                            int_cols.append(fname)

                for hint in _TEXT_HINTS:
                    for col in str_cols:
                        if hint in col.lower():
                            text_field = col
                            break
                    if text_field:
                        break
                if not text_field and str_cols:
                    text_field = str_cols[0]

                if classlabel_cols:
                    label_field = classlabel_cols[0]
                    ftype = features[label_field]
                    label_map = {i: n for i, n in enumerate(ftype.names)}
                else:
                    for hint in _LABEL_HINTS:
                        for col in int_cols:
                            if hint in col.lower():
                                label_field = col
                                break
                        if label_field:
                            break

                if not text_field or not label_field:
                    logger.debug("  ✗ %s — no text+label pair found", cand.id)
                    continue

                logger.info(
                    "  ✓ %s  (text='%s', label='%s', classes=%s)",
                    cand.id,
                    text_field,
                    label_field,
                    list(label_map.values()) if label_map else "auto",
                )
                sources.append(
                    {
                        "type": "hf_dataset",
                        "name": cand.id,
                        "split": "train",
                        "text_field": text_field,
                        "label_field": label_field,
                        "label_map": label_map,
                        "max_samples": max_samples,
                    }
                )
                break
            except Exception as exc:
                logger.debug("  ✗ %s — %s", cand.id, exc)

        if not sources:
            checked = [c.id for c in candidates[:5]]
            raise RuntimeError(
                f"Не нашлось подходящего датасета (text + label) по '{topic}'. "
                f"Проверенные: {checked}"
            )

        sources.append(
            {
                "type": "scrape",
                "url": "https://quotes.toscrape.com",
                "selector": "div.quote span.text",
                "max_pages": 5,
                "auto_label": True,
            }
        )

        slug = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")
        out_cfg = {"path": f"data/raw/{slug}_dataset.csv", "format": "csv"}

        return sources, out_cfg

    # ── skills ─────────────────────────────────────────────────────────

    def scrape(
        self,
        url: str,
        selector: str,
        max_pages: int = 1,
        auto_label: bool = False,
    ) -> pd.DataFrame:
        """Scrape text elements from *url* matching a CSS *selector*."""
        records: list[dict] = []
        now = datetime.now(timezone.utc).isoformat()

        for page in range(1, max_pages + 1):
            page_url = url if page == 1 else f"{url}/page/{page}/"
            try:
                resp = requests.get(
                    page_url,
                    headers={"User-Agent": "DataCollectionAgent/1.0"},
                    timeout=15,
                )
                resp.raise_for_status()
            except requests.RequestException as exc:
                logger.warning("GET %s → %s", page_url, exc)
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            elements = soup.select(selector)
            for el in elements:
                text = el.get_text(strip=True).strip("\u201c\u201d\"'")
                if not text:
                    continue
                label = _simple_sentiment(text) if auto_label else None
                records.append(
                    {
                        "text": text,
                        "label": label,
                        "source": f"scrape_{url}",
                        "collected_at": now,
                    }
                )
            logger.info("  page %d → %d elements", page, len(elements))

        return pd.DataFrame(records, columns=self.columns)

    def fetch_api(
        self,
        endpoint: str,
        params: dict | None = None,
        text_field: str = "text",
        label_field: str = "label",
        results_key: str | None = None,
        auto_label: bool = False,
    ) -> pd.DataFrame:
        """Fetch JSON data from a REST API *endpoint*."""
        now = datetime.now(timezone.utc).isoformat()
        try:
            resp = requests.get(
                endpoint,
                params=params,
                headers={"User-Agent": "DataCollectionAgent/1.0"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            logger.error("API %s → %s", endpoint, exc)
            return pd.DataFrame(columns=self.columns)

        if results_key:
            data = data.get(results_key, [])
        if not isinstance(data, list):
            data = [data]

        records: list[dict] = []
        for item in data:
            text = str(item.get(text_field, ""))
            raw_label = item.get(label_field)
            label = (
                raw_label
                if raw_label is not None
                else (_simple_sentiment(text) if auto_label else None)
            )
            records.append(
                {
                    "text": text,
                    "label": label,
                    "source": f"api_{endpoint}",
                    "collected_at": now,
                }
            )
        return pd.DataFrame(records, columns=self.columns)

    def load_dataset(
        self,
        name: str,
        source: str = "hf",
        split: str = "train",
        text_field: str = "text",
        label_field: str = "label",
        label_map: dict | None = None,
        max_samples: int | None = None,
    ) -> pd.DataFrame:
        """Load a dataset from HuggingFace Hub (``source='hf'``)."""
        now = datetime.now(timezone.utc).isoformat()

        if source != "hf":
            raise ValueError(f"Unsupported dataset source: {source}")

        cache_dir = os.environ.get("HF_HOME", os.path.join(os.getcwd(), ".hf_cache"))
        ds = hf_load_dataset(name, split=split, cache_dir=cache_dir)
        ds = ds.shuffle(seed=42)
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        df = ds.to_pandas()

        result = pd.DataFrame()
        result["text"] = df[text_field].astype(str)
        result["label"] = df[label_field]
        if label_map:
            result["label"] = result["label"].map(label_map).fillna(result["label"])
        result["source"] = f"{source}_{name}"
        result["collected_at"] = now

        return result[self.columns]

    def merge(self, sources: list[pd.DataFrame]) -> pd.DataFrame:
        """Concatenate, deduplicate by text, drop blanks."""
        if not sources:
            return pd.DataFrame(columns=self.columns)
        merged = pd.concat(sources, ignore_index=True)
        merged = merged.drop_duplicates(subset=["text"], keep="first")
        merged = merged.dropna(subset=["text"])
        merged = merged[merged["text"].str.strip().astype(bool)]
        return merged.reset_index(drop=True)

    # ── main entry point ───────────────────────────────────────────────

    def run(
        self,
        task: str | None = None,
        topic: str | None = None,
        sources: list[dict] | None = None,
    ) -> pd.DataFrame:
        """Collect from every source and return a unified DataFrame.

        Parameters
        ----------
        task : str, optional
            Name of a predefined task from the ``tasks`` section of the config.
        topic : str, optional
            Free-text description of the ML task.  The agent will search
            HuggingFace Hub for a matching dataset automatically.
        sources : list[dict], optional
            Explicit list of source descriptors (overrides *task* / *topic*).
        """
        out_cfg: dict = {}

        if topic is not None and sources is None:
            logger.info("═══ Topic: %s ═══", topic)
            sources, out_cfg = self.discover_sources(topic)
        elif task is not None:
            task_cfg = self._resolve_task(task)
            logger.info(
                "═══ Task: %s — %s ═══", task, task_cfg.get("description", "")
            )
            if sources is None:
                sources = task_cfg.get("sources", [])
            out_cfg = task_cfg.get("output", {})

        if sources is None:
            sources = self.config.get("sources", [])
        if not out_cfg:
            out_cfg = self.config.get("output", {})

        frames: list[pd.DataFrame] = []

        for src in sources:
            kind = src.get("type", "")
            tag = src.get("name", src.get("url", src.get("endpoint", "")))
            logger.info("Collecting [%s] %s …", kind, tag)

            try:
                if kind == "hf_dataset":
                    df = self.load_dataset(
                        name=src["name"],
                        source="hf",
                        split=src.get("split", "train"),
                        text_field=src.get("text_field", "text"),
                        label_field=src.get("label_field", "label"),
                        label_map=src.get("label_map"),
                        max_samples=src.get("max_samples"),
                    )
                elif kind == "scrape":
                    df = self.scrape(
                        url=src["url"],
                        selector=src["selector"],
                        max_pages=src.get("max_pages", 1),
                        auto_label=src.get("auto_label", False),
                    )
                elif kind == "api":
                    df = self.fetch_api(
                        endpoint=src["endpoint"],
                        params=src.get("params"),
                        text_field=src.get("text_field", "text"),
                        label_field=src.get("label_field", "label"),
                        results_key=src.get("results_key"),
                        auto_label=src.get("auto_label", False),
                    )
                else:
                    logger.warning("Unknown source type '%s' — skipping", kind)
                    continue

                logger.info("  → %d records", len(df))
                frames.append(df)

            except Exception:
                logger.exception("Failed to collect [%s] %s", kind, tag)

        result = self.merge(frames)
        logger.info("Merged dataset: %d records", len(result))

        if out_cfg.get("path"):
            os.makedirs(os.path.dirname(out_cfg["path"]) or ".", exist_ok=True)
            fmt = out_cfg.get("format", "csv")
            if fmt == "csv":
                result.to_csv(out_cfg["path"], index=False)
            elif fmt == "parquet":
                result.to_parquet(out_cfg["path"], index=False)
            logger.info("Saved → %s (%s)", out_cfg["path"], fmt)

        return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="DataCollectionAgent — сбор данных из нескольких источников",
    )
    parser.add_argument(
        "--task", "-t",
        help="Имя задачи из config.yaml (например: sentiment_analysis)",
    )
    parser.add_argument(
        "--topic", "-T",
        help="Произвольная тема на естественном языке (например: 'toxic comment classification')",
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Путь к конфигурационному файлу (по умолчанию: config.yaml)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        dest="list_tasks",
        help="Показать все доступные задачи и выйти",
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int,
        default=3000,
        help="Макс. кол-во записей из HF-датасета (по умолчанию: 3000)",
    )
    args = parser.parse_args()

    agent = DataCollectionAgent(config=args.config)

    if args.list_tasks:
        tasks = agent.available_tasks()
        if not tasks:
            print("В конфиге нет ни одной задачи.")
            return
        print("Доступные задачи:\n")
        for name, desc in tasks.items():
            print(f"  • {name:30s} {desc}")
        print(f"\nЗапуск:  python {__file__} --task <имя_задачи>")
        print(f"         python {__file__} --topic 'your topic in natural language'")
        return

    if not args.task and not args.topic:
        tasks = agent.available_tasks()
        print("Укажите задачу (--task) или тему (--topic).\n")
        if tasks:
            print("Готовые задачи из config.yaml:\n")
            for name, desc in tasks.items():
                print(f"  • {name:30s} {desc}")
        print(f"\nПримеры:")
        print(f"  python {__file__} --task sentiment_analysis")
        print(f"  python {__file__} --topic 'toxic comment classification'")
        print(f"  python {__file__} --topic 'emotion detection in tweets'")
        return

    df = agent.run(task=args.task, topic=args.topic)

    print(f"\nDataset shape: {df.shape}")
    print(df.head(10))
    print(f"\nLabel distribution:\n{df['label'].value_counts()}")
    print(f"\nSource distribution:\n{df['source'].value_counts()}")


if __name__ == "__main__":
    _cli()
