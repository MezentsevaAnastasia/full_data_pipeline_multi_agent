"""DataQualityAgent — автоматическое выявление и устранение проблем качества данных.

Skills:
  - detect_issues(df) → QualityReport
  - fix(df, strategy)  → DataFrame
  - compare(df_before, df_after) → ComparisonReport
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _iqr_bounds(series: pd.Series, k: float = 1.5) -> tuple[float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def _zscore_bounds(series: pd.Series, threshold: float = 3.0) -> tuple[float, float]:
    mean, std = series.mean(), series.std()
    return mean - threshold * std, mean + threshold * std


# ══════════════════════════════════════════════════════════════════════
# Agent
# ══════════════════════════════════════════════════════════════════════

class DataQualityAgent:
    """Агент-детектив для автоматического анализа и чистки данных.

    Parameters
    ----------
    outlier_method : str
        ``'iqr'`` (по умолчанию) или ``'zscore'``.
    iqr_k : float
        Коэффициент для IQR-метода (по умолчанию 1.5).
    zscore_threshold : float
        Порог z-score (по умолчанию 3.0).
    imbalance_threshold : float
        Порог отношения мин/макс класса для обнаружения дисбаланса (по умолчанию 0.7).
    """

    def __init__(
        self,
        outlier_method: str = "iqr",
        iqr_k: float = 1.5,
        zscore_threshold: float = 3.0,
        imbalance_threshold: float = 0.7,
    ):
        self.outlier_method = outlier_method
        self.iqr_k = iqr_k
        self.zscore_threshold = zscore_threshold
        self.imbalance_threshold = imbalance_threshold

    # ── skill: detect_issues ──────────────────────────────────────────

    def detect_issues(self, df: pd.DataFrame) -> dict[str, Any]:
        """Обнаружить проблемы качества данных.

        Returns
        -------
        dict с ключами ``missing``, ``duplicates``, ``outliers``, ``imbalance``.
        """
        report: dict[str, Any] = {}
        n = len(df)

        # 1. Missing values
        missing_info: dict[str, dict] = {}
        for col in df.columns:
            null_count = int(df[col].isna().sum())
            if is_string_dtype(df[col]):
                empty_str = int((df[col].fillna("").str.strip() == "").sum())
                null_count = max(null_count, empty_str)
            if null_count > 0:
                missing_info[col] = {
                    "count": null_count,
                    "percent": round(null_count / n * 100, 2),
                }
        report["missing"] = missing_info
        logger.info(
            "Missing values: %d columns affected",
            len(missing_info),
        )

        # 2. Duplicates
        dup_count = int(df.duplicated().sum())
        report["duplicates"] = {
            "count": dup_count,
            "percent": round(dup_count / n * 100, 2) if n else 0,
        }
        logger.info("Duplicates: %d (%.1f%%)", dup_count, report["duplicates"]["percent"])

        # 3. Outliers (numeric + text length)
        outlier_details: list[dict] = []

        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        text_cols = [
            c for c in df.columns
            if is_string_dtype(df[c]) and df[c].dropna().str.len().median() > 10
        ]
        synth_lengths: dict[str, pd.Series] = {}
        for col in text_cols:
            length_series = df[col].dropna().str.len().astype(float)
            synth_lengths[col] = length_series

        all_numeric = {c: df[c].dropna().astype(float) for c in numeric_cols}
        all_numeric.update({f"{c}_length": s for c, s in synth_lengths.items()})

        for col_name, series in all_numeric.items():
            if series.nunique() < 3:
                continue
            if self.outlier_method == "iqr":
                lo, hi = _iqr_bounds(series, self.iqr_k)
            else:
                lo, hi = _zscore_bounds(series, self.zscore_threshold)

            mask = (series < lo) | (series > hi)
            out_count = int(mask.sum())
            if out_count > 0:
                outlier_details.append({
                    "column": col_name,
                    "method": self.outlier_method,
                    "count": out_count,
                    "percent": round(out_count / len(series) * 100, 2),
                    "lower_bound": round(float(lo), 4),
                    "upper_bound": round(float(hi), 4),
                })
        report["outliers"] = outlier_details
        logger.info("Outliers: %d numeric/text-length features checked", len(all_numeric))

        # 4. Class imbalance
        categorical_cols = [c for c in df.columns if is_string_dtype(df[c]) or df[c].dtype.name == "category"]
        label_col = None
        for candidate in ("label", "target", "class", "category"):
            if candidate in df.columns:
                label_col = candidate
                break
        if label_col is None and len(categorical_cols) > 0:
            label_col = categorical_cols[0]

        imbalance_info: dict[str, Any] = {}
        if label_col is not None:
            counts = df[label_col].value_counts()
            ratio = counts.min() / counts.max() if counts.max() > 0 else 1.0
            imbalance_info = {
                "column": label_col,
                "distribution": counts.to_dict(),
                "imbalance_ratio": round(float(ratio), 4),
                "is_imbalanced": ratio < self.imbalance_threshold,
            }
            logger.info(
                "Imbalance (%s): ratio=%.3f, imbalanced=%s",
                label_col,
                ratio,
                imbalance_info["is_imbalanced"],
            )
        report["imbalance"] = imbalance_info

        report["shape"] = {"rows": n, "columns": len(df.columns)}
        return report

    # ── skill: fix ────────────────────────────────────────────────────

    def fix(
        self,
        df: pd.DataFrame,
        strategy: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Применить стратегии чистки.

        Parameters
        ----------
        strategy : dict
            Ключи: ``'missing'``, ``'duplicates'``, ``'outliers'``, ``'imbalance'``.

            missing:    ``'drop'`` | ``'median'`` | ``'mode'`` | ``'ffill'`` | ``'constant'``
            duplicates: ``'drop'``
            outliers:   ``'clip_iqr'`` | ``'clip_zscore'`` | ``'drop_iqr'`` | ``'drop_zscore'``
            imbalance:  ``'oversample'`` | ``'undersample'``
        """
        if strategy is None:
            strategy = {
                "missing": "drop",
                "duplicates": "drop",
                "outliers": "clip_iqr",
            }

        result = df.copy()
        logger.info("Fixing — strategy: %s", strategy)

        # 1. Duplicates
        dup_strat = strategy.get("duplicates", "")
        if dup_strat == "drop":
            before = len(result)
            result = result.drop_duplicates().reset_index(drop=True)
            logger.info("  duplicates/drop: %d → %d rows", before, len(result))

        # 2. Missing values
        miss_strat = strategy.get("missing", "")
        if miss_strat == "drop":
            before = len(result)
            result = result.dropna().reset_index(drop=True)
            logger.info("  missing/drop: %d → %d rows", before, len(result))
        elif miss_strat == "median":
            for col in result.columns:
                if is_numeric_dtype(result[col]):
                    result[col] = result[col].fillna(result[col].median())
                elif is_string_dtype(result[col]):
                    mode = result[col].mode()
                    result[col] = result[col].fillna(mode.iloc[0] if not mode.empty else "")
            logger.info("  missing/median: filled numeric with median, categorical with mode")
        elif miss_strat == "mode":
            for col in result.columns:
                if result[col].isna().any():
                    mode_val = result[col].mode()
                    result[col] = result[col].fillna(mode_val.iloc[0] if not mode_val.empty else "")
            logger.info("  missing/mode: filled all with mode")
        elif miss_strat == "ffill":
            result = result.ffill().bfill()
            logger.info("  missing/ffill: forward + backward fill")
        elif miss_strat == "constant":
            for col in result.columns:
                if is_numeric_dtype(result[col]):
                    result[col] = result[col].fillna(0)
                elif is_string_dtype(result[col]):
                    result[col] = result[col].fillna("UNKNOWN")
            logger.info("  missing/constant: 0 for numeric, 'UNKNOWN' for text")

        # 3. Outliers
        out_strat = strategy.get("outliers", "")
        if out_strat in ("clip_iqr", "clip_zscore", "drop_iqr", "drop_zscore"):
            method = "iqr" if "iqr" in out_strat else "zscore"
            action = "clip" if "clip" in out_strat else "drop"

            numeric_cols = result.select_dtypes(include="number").columns
            text_cols = [
                c for c in result.columns
                if is_string_dtype(result[c]) and result[c].dropna().str.len().median() > 10
            ]

            targets: list[tuple[str, pd.Series, str | None]] = []
            for c in numeric_cols:
                targets.append((c, result[c], None))
            for c in text_cols:
                targets.append((f"{c}_length", result[c].str.len(), c))

            rows_to_drop: set[int] = set()
            for col_name, series, source_col in targets:
                valid = series.dropna()
                if valid.nunique() < 3:
                    continue
                if method == "iqr":
                    lo, hi = _iqr_bounds(valid, self.iqr_k)
                else:
                    lo, hi = _zscore_bounds(valid, self.zscore_threshold)

                if action == "clip":
                    if source_col is None:
                        result[col_name] = series.clip(lower=lo, upper=hi)
                    # text length clipping — trim long texts
                    else:
                        max_len = int(hi)
                        mask = result[source_col].str.len() > max_len
                        result.loc[mask, source_col] = result.loc[mask, source_col].str[:max_len]
                else:
                    if source_col is None:
                        bad = series[(series < lo) | (series > hi)].index
                    else:
                        lengths = result[source_col].str.len()
                        bad = lengths[(lengths < lo) | (lengths > hi)].index
                    rows_to_drop.update(bad)

            if action == "drop" and rows_to_drop:
                before = len(result)
                result = result.drop(index=list(rows_to_drop)).reset_index(drop=True)
                logger.info("  outliers/%s_%s: %d → %d rows", action, method, before, len(result))
            elif action == "clip":
                logger.info("  outliers/clip_%s: values clipped to bounds", method)

        # 4. Imbalance
        imb_strat = strategy.get("imbalance", "")
        if imb_strat in ("oversample", "undersample"):
            label_col = None
            for candidate in ("label", "target", "class", "category"):
                if candidate in result.columns:
                    label_col = candidate
                    break
            if label_col:
                counts = result[label_col].value_counts()
                if imb_strat == "oversample":
                    max_count = counts.max()
                    frames = []
                    for cls, cnt in counts.items():
                        cls_df = result[result[label_col] == cls]
                        if cnt < max_count:
                            cls_df = cls_df.sample(max_count, replace=True, random_state=42)
                        frames.append(cls_df)
                    result = pd.concat(frames, ignore_index=True)
                    logger.info("  imbalance/oversample: all classes → %d each", max_count)
                else:
                    min_count = counts.min()
                    frames = []
                    for cls in counts.index:
                        frames.append(
                            result[result[label_col] == cls].sample(
                                min_count, random_state=42
                            )
                        )
                    result = pd.concat(frames, ignore_index=True)
                    logger.info("  imbalance/undersample: all classes → %d each", min_count)

        logger.info("Fix complete: %d rows × %d cols", *result.shape)
        return result

    # ── skill: compare ────────────────────────────────────────────────

    def compare(
        self,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
    ) -> pd.DataFrame:
        """Сравнительный отчёт «было / стало» по метрикам качества.

        Returns
        -------
        pd.DataFrame с колонками ``metric``, ``before``, ``after``, ``delta``.
        """
        report_before = self.detect_issues(df_before)
        report_after = self.detect_issues(df_after)

        rows: list[dict[str, Any]] = []

        rows.append({
            "metric": "rows",
            "before": report_before["shape"]["rows"],
            "after": report_after["shape"]["rows"],
        })
        rows.append({
            "metric": "columns",
            "before": report_before["shape"]["columns"],
            "after": report_after["shape"]["columns"],
        })

        # Missing
        total_miss_before = sum(v["count"] for v in report_before["missing"].values())
        total_miss_after = sum(v["count"] for v in report_after["missing"].values())
        rows.append({
            "metric": "missing_values_total",
            "before": total_miss_before,
            "after": total_miss_after,
        })
        all_cols = set(report_before["missing"]) | set(report_after["missing"])
        for col in sorted(all_cols):
            rows.append({
                "metric": f"missing_{col}",
                "before": report_before["missing"].get(col, {}).get("count", 0),
                "after": report_after["missing"].get(col, {}).get("count", 0),
            })

        # Duplicates
        rows.append({
            "metric": "duplicates",
            "before": report_before["duplicates"]["count"],
            "after": report_after["duplicates"]["count"],
        })

        # Outliers
        out_before = sum(o["count"] for o in report_before["outliers"])
        out_after = sum(o["count"] for o in report_after["outliers"])
        rows.append({
            "metric": "outliers_total",
            "before": out_before,
            "after": out_after,
        })

        # Imbalance ratio
        imb_before = report_before["imbalance"].get("imbalance_ratio", None)
        imb_after = report_after["imbalance"].get("imbalance_ratio", None)
        if imb_before is not None:
            rows.append({
                "metric": "imbalance_ratio",
                "before": imb_before,
                "after": imb_after,
            })

        comparison = pd.DataFrame(rows)
        comparison["delta"] = comparison["after"] - comparison["before"]
        comparison["delta_pct"] = np.where(
            comparison["before"] != 0,
            ((comparison["after"] - comparison["before"]) / comparison["before"] * 100).round(1),
            0,
        )

        return comparison


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="DataQualityAgent — детектив данных",
    )
    parser.add_argument("input", help="Путь к CSV-файлу с данными")
    parser.add_argument(
        "--strategy",
        default="missing=median,duplicates=drop,outliers=clip_iqr",
        help="Стратегия чистки (ключ=значение через запятую)",
    )
    parser.add_argument("--output", "-o", help="Сохранить очищенный датасет в CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    agent = DataQualityAgent()

    print("\n══════ Quality Report ══════")
    report = agent.detect_issues(df)
    for key, val in report.items():
        print(f"\n  {key}: {val}")

    strategy = dict(kv.split("=") for kv in args.strategy.split(","))
    df_clean = agent.fix(df, strategy=strategy)

    print("\n══════ Comparison ══════")
    comp = agent.compare(df, df_clean)
    print(comp.to_string(index=False))

    if args.output:
        df_clean.to_csv(args.output, index=False)
        print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    _cli()
