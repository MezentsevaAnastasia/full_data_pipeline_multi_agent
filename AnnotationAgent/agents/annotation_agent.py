"""AnnotationAgent — автоматическая разметка данных, генерация спецификации и экспорт.

Skills:
  - auto_label(df)            → DataFrame с метками и confidence
  - generate_spec(df, task)   → str (путь к Markdown-файлу)
  - check_quality(df_labeled) → dict с метриками качества
  - export_to_labelstudio(df) → str (путь к JSON)
"""

from __future__ import annotations

import os

# Apple Silicon: prevent SIGBUS alignment fault in Accelerate BLAS
# during multi-threaded SGEMM inside PyTorch. Must be set before torch loads.
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

_TEXT_COL_CANDIDATES = ("text", "sentence", "review", "comment", "content", "body", "message")
_LABEL_COL_CANDIDATES = ("label", "target", "class", "category", "sentiment")


def _find_text_column(df: pd.DataFrame) -> str | None:
    """Эвристика для поиска текстовой колонки."""
    for candidate in _TEXT_COL_CANDIDATES:
        if candidate in df.columns:
            return candidate
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            if df[col].dropna().str.len().median() > 20:
                return col
    return None


def _find_label_column(df: pd.DataFrame) -> str | None:
    for candidate in _LABEL_COL_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


# ══════════════════════════════════════════════════════════════════════
# Agent
# ══════════════════════════════════════════════════════════════════════

class AnnotationAgent:
    """Агент для автоматической разметки данных.

    Parameters
    ----------
    modality : str
        Модальность данных: ``'text'``, ``'audio'`` или ``'image'``.
    model_name : str | None
        Название модели для zero-shot классификации.
        По умолчанию ``'facebook/bart-large-mnli'`` для text.
    candidate_labels : list[str] | None
        Список меток для zero-shot классификации.
        Если None — определяется автоматически из данных.
    confidence_threshold : float
        Порог уверенности для HITL-флагирования (по умолчанию 0.7).
    batch_size : int
        Размер батча для инференса.
    """

    _DEFAULT_MODELS = {
        "text": "facebook/bart-large-mnli",
        "audio": "openai/whisper-base",
        "image": "openai/clip-vit-base-patch32",
    }

    def __init__(
        self,
        modality: str = "text",
        model_name: str | None = None,
        candidate_labels: list[str] | None = None,
        confidence_threshold: float = 0.7,
        batch_size: int = 32,
    ):
        if modality not in self._DEFAULT_MODELS:
            raise ValueError(
                f"Unsupported modality: {modality!r}. "
                f"Choose from {list(self._DEFAULT_MODELS)}."
            )

        self.modality = modality
        self.model_name = model_name or self._DEFAULT_MODELS[modality]
        self.candidate_labels = candidate_labels
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self._pipeline = None

    def _get_pipeline(self):
        """Ленивая инициализация HuggingFace pipeline."""
        if self._pipeline is not None:
            return self._pipeline

        import torch
        from transformers import pipeline as hf_pipeline

        torch.set_num_threads(1)

        # Apple Silicon (ARM64): Accelerate's SGEMM has an alignment fault bug
        # with memory-mapped float32 tensors. Using float16 routes through HGEMM
        # which does not have this issue.
        use_fp16 = torch.backends.mps.is_available() or (
            not torch.cuda.is_available() and torch.get_default_dtype() == torch.float32
        )
        dtype = torch.float16 if use_fp16 else None

        logger.info("Loading model %s (dtype=%s) …", self.model_name, dtype)

        if self.modality == "text":
            self._pipeline = hf_pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device="cpu",
                torch_dtype=dtype,
            )
        else:
            raise NotImplementedError(
                f"Pipeline for modality '{self.modality}' not yet implemented."
            )
        return self._pipeline

    # ── skill: auto_label ─────────────────────────────────────────────

    def auto_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Автоматически разметить данные с помощью zero-shot классификации.

        Parameters
        ----------
        df : pd.DataFrame
            Входной датафрейм. Должен содержать текстовую колонку.

        Returns
        -------
        pd.DataFrame
            Копия с колонками ``predicted_label``, ``confidence``, ``needs_review``.
        """
        if self.modality != "text":
            raise NotImplementedError(
                f"auto_label for '{self.modality}' not yet implemented."
            )

        text_col = _find_text_column(df)
        if text_col is None:
            raise ValueError("Cannot find text column in DataFrame.")

        labels = self.candidate_labels
        if labels is None:
            label_col = _find_label_column(df)
            if label_col is not None:
                labels = sorted(df[label_col].dropna().unique().tolist())
                logger.info("Auto-detected labels from '%s': %s", label_col, labels)
            else:
                labels = ["positive", "negative"]
                logger.info("No label column found, defaulting to: %s", labels)

        pipe = self._get_pipeline()
        texts = df[text_col].fillna("").tolist()

        predicted_labels: list[str] = []
        confidences: list[float] = []

        total = len(texts)
        for start in range(0, total, self.batch_size):
            batch = texts[start : start + self.batch_size]
            results = pipe(batch, candidate_labels=labels, truncation=True)
            if isinstance(results, dict):
                results = [results]
            for r in results:
                predicted_labels.append(r["labels"][0])
                confidences.append(round(r["scores"][0], 4))
            logger.info(
                "  auto_label progress: %d / %d",
                min(start + self.batch_size, total),
                total,
            )

        result = df.copy()
        result["predicted_label"] = predicted_labels
        result["confidence"] = confidences
        result["needs_review"] = result["confidence"] < self.confidence_threshold

        n_review = int(result["needs_review"].sum())
        logger.info(
            "auto_label complete: %d rows, %d flagged for review (confidence < %.2f)",
            len(result), n_review, self.confidence_threshold,
        )
        return result

    # ── skill: generate_spec ──────────────────────────────────────────

    def generate_spec(
        self,
        df: pd.DataFrame,
        task: str = "classification",
        output_path: str = "annotation_spec.md",
    ) -> str:
        """Сгенерировать спецификацию разметки (Markdown-файл).

        Parameters
        ----------
        df : pd.DataFrame
            Датафрейм с колонкой меток (``predicted_label`` или ``label``).
        task : str
            Название задачи (например, ``'sentiment_classification'``).
        output_path : str
            Путь для сохранения файла.

        Returns
        -------
        str
            Путь к сгенерированному файлу.
        """
        label_col = None
        for candidate in ("predicted_label", "label", "target", "class", "category"):
            if candidate in df.columns:
                label_col = candidate
                break
        if label_col is None:
            raise ValueError("No label column found in DataFrame.")

        text_col = _find_text_column(df)
        classes = sorted(df[label_col].dropna().unique().tolist())
        class_defs = self._class_definitions(task, classes)

        lines: list[str] = [
            f"# Annotation Specification: {task}",
            "",
            "## Задача",
            "",
            f"- **Тип задачи:** {task}",
            f"- **Модальность:** {self.modality}",
            f"- **Количество классов:** {len(classes)}",
            f"- **Объём данных:** {len(df)} примеров",
            "",
            "---",
            "",
            "## Классы",
            "",
        ]

        for cls in classes:
            lines.append(f"### {cls}")
            lines.append("")
            lines.append(f"**Определение:** {class_defs.get(cls, f'Класс *{cls}*.')}")
            lines.append("")

            cls_df = df[df[label_col] == cls]
            if text_col is not None and len(cls_df) > 0:
                n_examples = min(3, len(cls_df))
                samples = cls_df.sample(n=n_examples, random_state=42)
                lines.append("**Примеры:**")
                lines.append("")
                for i, (_, row) in enumerate(samples.iterrows(), 1):
                    text = str(row[text_col])[:200]
                    lines.append(f'{i}. "{text}"')
                lines.append("")

        lines.extend([
            "---",
            "",
            "## Граничные случаи",
            "",
        ])
        for i, case in enumerate(
            self._edge_cases(task, classes, df, text_col, label_col), 1
        ):
            lines.append(f"{i}. {case}")

        lines.extend([
            "",
            "---",
            "",
            "## Инструкции для разметчика",
            "",
            "1. Прочитайте текст полностью перед принятием решения.",
            "2. Выберите **один** класс, наиболее подходящий к тексту.",
            "3. Если текст неоднозначный — отметьте как требующий обсуждения.",
            "4. Если текст не относится ни к одному классу — пропустите.",
            "5. Обращайте внимание на сарказм и иронию.",
            "",
        ])

        spec_text = "\n".join(lines)
        Path(output_path).write_text(spec_text, encoding="utf-8")
        logger.info("Annotation spec saved → %s", output_path)
        return output_path

    # ── skill: check_quality ──────────────────────────────────────────

    def check_quality(self, df_labeled: pd.DataFrame) -> dict[str, Any]:
        """Оценить качество разметки.

        Parameters
        ----------
        df_labeled : pd.DataFrame
            Датафрейм с ``predicted_label``, ``confidence`` и (опц.) ``label``.

        Returns
        -------
        dict
            ``kappa``, ``agreement``, ``label_dist``, ``confidence_mean``,
            ``confidence_std``, ``low_confidence_count``, ``low_confidence_pct``.
        """
        metrics: dict[str, Any] = {}

        pred_col = (
            "predicted_label" if "predicted_label" in df_labeled.columns
            else _find_label_column(df_labeled)
        )
        if pred_col is None:
            raise ValueError("No label column found.")

        # 1. Label distribution
        metrics["label_dist"] = df_labeled[pred_col].value_counts().to_dict()
        logger.info("Label distribution: %s", metrics["label_dist"])

        # 2. Confidence statistics
        if "confidence" in df_labeled.columns:
            conf = df_labeled["confidence"]
            metrics["confidence_mean"] = round(float(conf.mean()), 4)
            metrics["confidence_std"] = round(float(conf.std()), 4)
            metrics["confidence_min"] = round(float(conf.min()), 4)
            metrics["confidence_max"] = round(float(conf.max()), 4)

            n_low = int((conf < self.confidence_threshold).sum())
            metrics["low_confidence_count"] = n_low
            metrics["low_confidence_pct"] = round(n_low / len(df_labeled) * 100, 2)
            logger.info(
                "Confidence: mean=%.3f, std=%.3f, low_conf=%d (%.1f%%)",
                metrics["confidence_mean"], metrics["confidence_std"],
                n_low, metrics["low_confidence_pct"],
            )

        # 3. Cohen's κ (auto vs ground truth)
        gt_col = None
        if "label" in df_labeled.columns and pred_col == "predicted_label":
            gt_col = "label"
        elif "ground_truth" in df_labeled.columns:
            gt_col = "ground_truth"

        if gt_col is not None:
            mask = df_labeled[gt_col].notna() & df_labeled[pred_col].notna()
            if mask.sum() > 0:
                kappa = cohen_kappa_score(
                    df_labeled.loc[mask, gt_col],
                    df_labeled.loc[mask, pred_col],
                )
                agreement = float(
                    (df_labeled.loc[mask, gt_col] == df_labeled.loc[mask, pred_col]).mean()
                )
                metrics["kappa"] = round(float(kappa), 4)
                metrics["agreement"] = round(agreement, 4)
                logger.info(
                    "Cohen's κ = %.4f, agreement = %.2f%%", kappa, agreement * 100,
                )
        else:
            metrics["kappa"] = None
            metrics["agreement"] = None
            logger.info("No ground truth column — κ not computed.")

        return metrics

    # ── skill: export_to_labelstudio ──────────────────────────────────

    def export_to_labelstudio(
        self,
        df: pd.DataFrame,
        output_path: str = "labelstudio_import.json",
    ) -> str:
        """Экспортировать данные в формат LabelStudio import.

        Parameters
        ----------
        df : pd.DataFrame
            Датафрейм с текстами и (опционально) метками.
        output_path : str
            Путь к выходному JSON-файлу.

        Returns
        -------
        str
            Путь к сохранённому файлу.
        """
        text_col = _find_text_column(df)
        if text_col is None:
            raise ValueError("Cannot find text column in DataFrame.")

        label_col = None
        for candidate in ("predicted_label", "label", "target"):
            if candidate in df.columns:
                label_col = candidate
                break

        tasks: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            task_item: dict[str, Any] = {
                "data": {"text": str(row[text_col])},
            }

            if "source" in df.columns:
                task_item["data"]["source"] = str(row["source"])
            if "confidence" in df.columns:
                task_item["data"]["confidence"] = float(row["confidence"])

            if label_col is not None and pd.notna(row.get(label_col)):
                score = float(row["confidence"]) if "confidence" in df.columns else 1.0
                task_item["predictions"] = [
                    {
                        "model_version": self.model_name,
                        "result": [
                            {
                                "from_name": "label",
                                "to_name": "text",
                                "type": "choices",
                                "value": {"choices": [str(row[label_col])]},
                            }
                        ],
                        "score": score,
                    }
                ]

            tasks.append(task_item)

        Path(output_path).write_text(
            json.dumps(tasks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "LabelStudio export saved → %s (%d tasks)", output_path, len(tasks),
        )
        return output_path

    # ── bonus: HITL — flag low confidence ─────────────────────────────

    def flag_for_review(
        self,
        df_labeled: pd.DataFrame,
        output_path: str = "review_queue.csv",
    ) -> pd.DataFrame:
        """Выделить примеры с низкой уверенностью для ручной разметки (HITL).

        Parameters
        ----------
        df_labeled : pd.DataFrame
            Датафрейм после ``auto_label`` (с ``confidence``).
        output_path : str
            Путь к CSV-файлу для ручной проверки.

        Returns
        -------
        pd.DataFrame
            Подмножество с низкой уверенностью.
        """
        if "confidence" not in df_labeled.columns:
            raise ValueError("Column 'confidence' not found. Run auto_label first.")

        if "needs_review" in df_labeled.columns:
            review_df = df_labeled[df_labeled["needs_review"]].copy()
        else:
            review_df = df_labeled[
                df_labeled["confidence"] < self.confidence_threshold
            ].copy()

        review_df = review_df.sort_values("confidence").reset_index(drop=True)
        review_df.to_csv(output_path, index=False)
        logger.info(
            "Review queue saved → %s (%d examples, confidence < %.2f)",
            output_path, len(review_df), self.confidence_threshold,
        )
        return review_df

    # ── internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _class_definitions(task: str, classes: list[str]) -> dict[str, str]:
        _TEMPLATES: dict[str, dict[str, str]] = {
            "sentiment": {
                "positive": (
                    "Текст выражает положительное мнение, одобрение, "
                    "удовлетворение или похвалу."
                ),
                "negative": (
                    "Текст выражает отрицательное мнение, критику, "
                    "неудовольствие или разочарование."
                ),
                "neutral": (
                    "Текст не выражает явной эмоциональной окраски "
                    "или содержит сбалансированное мнение."
                ),
            },
            "spam": {
                "spam": "Нежелательная реклама, мошенничество или нерелевантный контент.",
                "ham": "Обычное, легитимное сообщение.",
                "not spam": "Обычное, легитимное сообщение.",
            },
            "toxic": {
                "toxic": "Текст содержит оскорбления, угрозы или ненависть.",
                "not toxic": "Текст не содержит вредоносного контента.",
                "non-toxic": "Текст не содержит вредоносного контента.",
            },
        }
        task_lower = task.lower()
        for key, mapping in _TEMPLATES.items():
            if key in task_lower:
                return {c: mapping.get(c.lower(), f"Класс *{c}*.") for c in classes}
        return {c: f"Пример относится к классу *{c}*." for c in classes}

    @staticmethod
    def _edge_cases(
        task: str,
        classes: list[str],
        df: pd.DataFrame,
        text_col: str | None,
        label_col: str,
    ) -> list[str]:
        cases: list[str] = []
        task_lower = task.lower()

        if "sentiment" in task_lower:
            cases.extend([
                "**Сарказм / ирония:** текст формально положительный, но по смыслу "
                "негативный. Размечайте по *истинному* смыслу.",
                "**Смешанное мнение:** текст содержит и плюсы, и минусы. "
                "Выберите доминирующий тон.",
                "**Нейтральные факты:** текст описывает факты без оценки. "
                "Если нет класса 'neutral' — выберите ближайший.",
            ])
        elif "spam" in task_lower:
            cases.extend([
                "**Рекламный, но полезный контент:** если сообщение полезно, "
                "но явно рекламное — размечайте как spam.",
                "**Системные уведомления:** автоматические уведомления без рекламы — ham.",
            ])
        elif "toxic" in task_lower:
            cases.extend([
                "**Цитирование:** если текст цитирует оскорбительный контент "
                "для обсуждения — не считать toxic.",
                "**Грубый юмор:** грубые шутки без прямых оскорблений — "
                "учитывайте контекст.",
            ])

        if text_col and "confidence" in df.columns:
            low_conf = df[df["confidence"] < 0.6]
            if len(low_conf) > 0:
                samples = low_conf.sample(n=min(2, len(low_conf)), random_state=42)
                for _, row in samples.iterrows():
                    preview = str(row[text_col])[:120]
                    cases.append(
                        f"**Пример с низкой уверенностью** "
                        f"(confidence={row['confidence']:.2f}): \"{preview}…\""
                    )

        if not cases:
            cases.append(
                "Если текст не подходит ни к одному классу — отметьте для обсуждения."
            )
            cases.append(
                "Очень короткие тексты (1-2 слова) могут быть неоднозначны — "
                "размечайте по контексту."
            )

        return cases


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="AnnotationAgent — автоматическая разметка данных",
    )
    parser.add_argument("input", help="Путь к CSV-файлу с данными")
    parser.add_argument(
        "--modality", default="text",
        choices=("text", "audio", "image"),
        help="Модальность данных",
    )
    parser.add_argument("--task", default="classification", help="Название задачи")
    parser.add_argument("--labels", nargs="+", help="Список меток для классификации")
    parser.add_argument(
        "--threshold", type=float, default=0.7,
        help="Порог confidence для HITL",
    )
    parser.add_argument("--output-dir", "-o", default=".", help="Директория для выходных файлов")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    agent = AnnotationAgent(
        modality=args.modality,
        candidate_labels=args.labels,
        confidence_threshold=args.threshold,
    )

    print("\n══════ Auto-labeling ══════")
    df_labeled = agent.auto_label(df)
    df_labeled.to_csv(out / "labeled_data.csv", index=False)
    print(f"  Labeled: {len(df_labeled)} rows")

    print("\n══════ Annotation Spec ══════")
    spec_path = agent.generate_spec(
        df_labeled, task=args.task,
        output_path=str(out / "annotation_spec.md"),
    )
    print(f"  Spec → {spec_path}")

    print("\n══════ Quality Metrics ══════")
    metrics = agent.check_quality(df_labeled)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\n══════ LabelStudio Export ══════")
    ls_path = agent.export_to_labelstudio(
        df_labeled, output_path=str(out / "labelstudio_import.json"),
    )
    print(f"  Export → {ls_path}")

    print("\n══════ HITL Review Queue ══════")
    review_df = agent.flag_for_review(
        df_labeled, output_path=str(out / "review_queue.csv"),
    )
    print(f"  Review queue: {len(review_df)} examples")


if __name__ == "__main__":
    _cli()
