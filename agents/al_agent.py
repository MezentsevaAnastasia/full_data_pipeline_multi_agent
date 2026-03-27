"""ActiveLearningAgent — умный отбор данных для разметки.

Skills:
  - fit(labeled_df)                      → fitted model
  - query(pool_df, strategy)             → indices
  - evaluate(labeled_df, test_df)        → dict (accuracy, f1)
  - report(history)                      → str  (путь к learning_curve.png)
  - run_cycle(labeled_df, pool_df, ...)  → list[dict]
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

_TEXT_COL_CANDIDATES = ("text", "sentence", "review", "comment", "content", "body", "message")
_LABEL_COL_CANDIDATES = ("label", "target", "class", "category", "sentiment")

Strategy = Literal["entropy", "margin", "random"]


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


class ActiveLearningAgent:
    """Агент для Active Learning — итеративного отбора наиболее информативных
    примеров из пула неразмеченных данных.

    Parameters
    ----------
    model : str
        Тип классификатора: ``'logreg'`` или ``'rf'`` (Random Forest).
    max_features : int
        Максимальное число TF-IDF-признаков.
    random_state : int
        Seed для воспроизводимости.
    """

    _MODELS = {
        "logreg": lambda rs: LogisticRegression(
            C=1000.0, max_iter=3000, solver="lbfgs", random_state=rs,
        ),
        "rf": lambda rs: RandomForestClassifier(
            n_estimators=200, random_state=rs,
        ),
    }

    def __init__(
        self,
        model: str = "logreg",
        max_features: int = 5000,
        n_components: int = 50,
        random_state: int = 42,
    ):
        if model not in self._MODELS:
            raise ValueError(
                f"Unknown model {model!r}. Choose from {list(self._MODELS)}."
            )
        self.model_name = model
        self.max_features = max_features
        self.n_components = n_components
        self.random_state = random_state

        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )
        self._svd = make_pipeline(
            TruncatedSVD(n_components=n_components, random_state=random_state),
            Normalizer(copy=False),
        )
        self._clf = self._MODELS[model](random_state)
        self._is_fitted = False
        self._vectorizer_fitted = False

    # ── unsupervised vocabulary ────────────────────────────────────────

    def build_vocabulary(self, all_texts_df: pd.DataFrame) -> "ActiveLearningAgent":
        """Построить TF-IDF + SVD представление на всех доступных текстах (unsupervised).

        Вызывается один раз перед AL-циклом, чтобы словарь и SVD-проекция
        покрывали весь корпус, а не только seed.
        """
        text_col = _find_column(all_texts_df, _TEXT_COL_CANDIDATES)
        if text_col is None:
            raise ValueError("Cannot find text column.")
        texts = all_texts_df[text_col].fillna("").values
        X_tfidf = self._vectorizer.fit_transform(texts)
        self._svd.fit(X_tfidf)
        self._vectorizer_fitted = True
        logger.info("build_vocabulary: %d documents, %d tfidf → %d svd features",
                     len(texts), X_tfidf.shape[1], self.n_components)
        return self

    def _featurize(self, texts: np.ndarray) -> np.ndarray:
        """TF-IDF → SVD → dense feature matrix."""
        X_tfidf = self._vectorizer.transform(texts)
        return self._svd.transform(X_tfidf)

    # ── skill: fit ─────────────────────────────────────────────────────

    def fit(self, labeled_df: pd.DataFrame) -> "ActiveLearningAgent":
        """Обучить модель на размеченных данных.

        Returns
        -------
        self
            Агент с обученной моделью (для chaining).
        """
        text_col = _find_column(labeled_df, _TEXT_COL_CANDIDATES)
        label_col = _find_column(labeled_df, _LABEL_COL_CANDIDATES)
        if text_col is None or label_col is None:
            raise ValueError(
                f"Cannot find text ({text_col}) or label ({label_col}) column."
            )

        texts = labeled_df[text_col].fillna("").values
        labels = labeled_df[label_col].values

        if not self._vectorizer_fitted:
            X_tfidf = self._vectorizer.fit_transform(texts)
            self._svd.fit(X_tfidf)
            self._vectorizer_fitted = True
            X = self._svd.transform(X_tfidf)
        else:
            X = self._featurize(texts)
        self._clf.fit(X, labels)
        self._is_fitted = True

        logger.info(
            "fit: trained %s on %d samples (%d features)",
            self.model_name, len(labels), X.shape[1],
        )
        return self

    # ── skill: query ───────────────────────────────────────────────────

    def query(
        self,
        pool_df: pd.DataFrame,
        strategy: Strategy = "entropy",
        batch_size: int = 20,
    ) -> np.ndarray:
        """Выбрать наиболее информативные примеры из пула.

        Parameters
        ----------
        pool_df : pd.DataFrame
            Неразмеченные данные.
        strategy : str
            ``'entropy'``, ``'margin'`` или ``'random'``.
        batch_size : int
            Количество примеров для отбора.

        Returns
        -------
        np.ndarray
            Индексы выбранных примеров (позиционные, от 0 до len(pool)-1).
        """
        n = len(pool_df)
        batch_size = min(batch_size, n)

        if strategy == "random" or not self._is_fitted:
            rng = np.random.RandomState(self.random_state)
            indices = rng.choice(n, size=batch_size, replace=False)
            logger.info("query/random: selected %d samples", batch_size)
            return indices

        text_col = _find_column(pool_df, _TEXT_COL_CANDIDATES)
        if text_col is None:
            raise ValueError("Cannot find text column in pool DataFrame.")

        texts = pool_df[text_col].fillna("").values
        X = self._featurize(texts)
        proba = self._clf.predict_proba(X)

        if strategy == "entropy":
            scores = -np.sum(proba * np.log(proba + 1e-12), axis=1)
        elif strategy == "margin":
            sorted_proba = np.sort(proba, axis=1)
            scores = -(sorted_proba[:, -1] - sorted_proba[:, -2])
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

        rng = np.random.RandomState(self.random_state + n)
        noise = rng.uniform(0, 1e-10, size=len(scores))
        scores = scores + noise

        indices = np.argsort(scores)[-batch_size:]
        logger.info(
            "query/%s: selected %d samples (score range: %.4f – %.4f)",
            strategy, batch_size,
            float(scores[indices[0]]), float(scores[indices[-1]]),
        )
        return indices

    # ── skill: evaluate ────────────────────────────────────────────────

    def evaluate(
        self,
        labeled_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> dict[str, float]:
        """Оценить модель на тестовом наборе.

        Returns
        -------
        dict
            ``accuracy`` и ``f1`` (macro-averaged).
        """
        if not self._is_fitted:
            self.fit(labeled_df)

        text_col = _find_column(test_df, _TEXT_COL_CANDIDATES)
        label_col = _find_column(test_df, _LABEL_COL_CANDIDATES)
        if text_col is None or label_col is None:
            raise ValueError("Cannot find text/label columns in test DataFrame.")

        texts = test_df[text_col].fillna("").values
        y_true = test_df[label_col].values

        X = self._featurize(texts)
        y_pred = self._clf.predict(X)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")

        logger.info("evaluate: accuracy=%.4f, f1=%.4f", acc, f1)
        return {"accuracy": round(float(acc), 4), "f1": round(float(f1), 4)}

    # ── skill: report ──────────────────────────────────────────────────

    def report(
        self,
        history: list[dict[str, Any]],
        output_path: str = "learning_curve.png",
        title: str | None = None,
    ) -> str:
        """Построить кривую обучения (quality vs n_labeled) и сохранить график.

        Parameters
        ----------
        history : list[dict]
            Список записей ``{iteration, n_labeled, accuracy, f1, strategy}``.
        output_path : str
            Путь к выходному файлу.
        title : str | None
            Заголовок графика.

        Returns
        -------
        str
            Путь к сохранённому файлу.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df_h = pd.DataFrame(history)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        strategies = df_h["strategy"].unique() if "strategy" in df_h.columns else ["default"]

        for strat in strategies:
            sub = df_h[df_h["strategy"] == strat] if "strategy" in df_h.columns else df_h
            axes[0].plot(sub["n_labeled"], sub["accuracy"], "o-", label=strat)
            axes[1].plot(sub["n_labeled"], sub["f1"], "o-", label=strat)

        axes[0].set_xlabel("Размеченных примеров")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Accuracy vs размер обучающей выборки")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel("Размеченных примеров")
        axes[1].set_ylabel("F1 (macro)")
        axes[1].set_title("F1 vs размер обучающей выборки")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        if title:
            fig.suptitle(title, fontsize=14, y=1.02)

        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("Learning curve saved → %s", output_path)
        return output_path

    # ── orchestrator: run_cycle ────────────────────────────────────────

    def run_cycle(
        self,
        labeled_df: pd.DataFrame,
        pool_df: pd.DataFrame,
        test_df: pd.DataFrame | None = None,
        strategy: Strategy = "entropy",
        n_iterations: int = 5,
        batch_size: int = 20,
    ) -> list[dict[str, Any]]:
        """Запустить полный цикл Active Learning.

        Parameters
        ----------
        labeled_df : pd.DataFrame
            Начальный размеченный набор (seed).
        pool_df : pd.DataFrame
            Пул неразмеченных данных (с метками, скрытыми от модели).
        test_df : pd.DataFrame | None
            Тестовый набор. Если None — 20% от pool_df используются как тест.
        strategy : str
            Стратегия отбора: ``'entropy'``, ``'margin'`` или ``'random'``.
        n_iterations : int
            Число итераций.
        batch_size : int
            Размер батча отбора.

        Returns
        -------
        list[dict]
            История: ``{iteration, n_labeled, accuracy, f1, strategy}``.
        """
        label_col = _find_column(labeled_df, _LABEL_COL_CANDIDATES)
        if label_col is None:
            raise ValueError("Cannot find label column in labeled_df.")

        if test_df is None:
            n_test = max(int(len(pool_df) * 0.2), 50)
            test_df = pool_df.sample(n=n_test, random_state=self.random_state)
            pool_df = pool_df.drop(test_df.index).reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)
            logger.info("No test_df provided; split %d test, %d pool", len(test_df), len(pool_df))

        current_labeled = labeled_df.copy().reset_index(drop=True)
        current_pool = pool_df.copy().reset_index(drop=True)
        history: list[dict[str, Any]] = []

        if not self._vectorizer_fitted:
            all_text = pd.concat([labeled_df, pool_df, test_df], ignore_index=True)
            self.build_vocabulary(all_text)

        self.fit(current_labeled)
        metrics = self.evaluate(current_labeled, test_df)
        history.append({
            "iteration": 0,
            "n_labeled": len(current_labeled),
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "strategy": strategy,
        })
        logger.info(
            "iter 0 (seed): n=%d, acc=%.4f, f1=%.4f",
            len(current_labeled), metrics["accuracy"], metrics["f1"],
        )

        for i in range(1, n_iterations + 1):
            if len(current_pool) == 0:
                logger.warning("Pool exhausted at iteration %d", i)
                break

            indices = self.query(current_pool, strategy=strategy, batch_size=batch_size)
            selected = current_pool.iloc[indices].copy()
            current_pool = current_pool.drop(current_pool.index[indices]).reset_index(drop=True)
            current_labeled = pd.concat(
                [current_labeled, selected], ignore_index=True,
            )

            self.fit(current_labeled)
            metrics = self.evaluate(current_labeled, test_df)

            history.append({
                "iteration": i,
                "n_labeled": len(current_labeled),
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "strategy": strategy,
            })
            logger.info(
                "iter %d: n=%d (+%d), acc=%.4f, f1=%.4f",
                i, len(current_labeled), len(selected),
                metrics["accuracy"], metrics["f1"],
            )

        return history

    # ── утилита: сравнить стратегии ────────────────────────────────────

    def compare_strategies(
        self,
        labeled_df: pd.DataFrame,
        pool_df: pd.DataFrame,
        test_df: pd.DataFrame | None = None,
        strategies: list[Strategy] | None = None,
        n_iterations: int = 5,
        batch_size: int = 20,
        output_path: str = "learning_curve.png",
    ) -> dict[str, list[dict[str, Any]]]:
        """Запустить run_cycle для нескольких стратегий и построить
        сравнительный график.

        Returns
        -------
        dict
            ``{strategy_name: history_list}``
        """
        if strategies is None:
            strategies = ["entropy", "random"]

        all_history: dict[str, list[dict[str, Any]]] = {}
        combined: list[dict[str, Any]] = []

        all_text = pd.concat(
            [labeled_df, pool_df] + ([test_df] if test_df is not None else []),
            ignore_index=True,
        )

        for strat in strategies:
            logger.info("═══ Strategy: %s ═══", strat)
            agent = ActiveLearningAgent(
                model=self.model_name,
                max_features=self.max_features,
                n_components=self.n_components,
                random_state=self.random_state,
            )
            agent.build_vocabulary(all_text)
            h = agent.run_cycle(
                labeled_df=labeled_df.copy(),
                pool_df=pool_df.copy(),
                test_df=test_df.copy() if test_df is not None else None,
                strategy=strat,
                n_iterations=n_iterations,
                batch_size=batch_size,
            )
            all_history[strat] = h
            combined.extend(h)

        self.report(
            combined,
            output_path=output_path,
            title="Active Learning: сравнение стратегий",
        )
        return all_history


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="ActiveLearningAgent — умный отбор данных",
    )
    parser.add_argument("input", help="Путь к CSV-файлу с данными")
    parser.add_argument(
        "--model", default="logreg", choices=("logreg", "rf"),
        help="Тип классификатора",
    )
    parser.add_argument("--seed-size", type=int, default=50, help="Начальный размер seed")
    parser.add_argument("--iterations", type=int, default=5, help="Число итераций AL")
    parser.add_argument("--batch-size", type=int, default=20, help="Размер батча")
    parser.add_argument(
        "--strategy", default="entropy",
        choices=("entropy", "margin", "random"),
        help="Стратегия отбора",
    )
    parser.add_argument("--compare", action="store_true", help="Сравнить entropy vs random")
    parser.add_argument("--output", "-o", default=".", help="Директория для выходных файлов")
    args = parser.parse_args()

    from pathlib import Path

    df = pd.read_csv(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)
    indices = rng.permutation(len(df))
    seed_idx = indices[: args.seed_size]
    pool_idx = indices[args.seed_size :]

    labeled_df = df.iloc[seed_idx].reset_index(drop=True)
    pool_df = df.iloc[pool_idx].reset_index(drop=True)

    agent = ActiveLearningAgent(model=args.model)

    if args.compare:
        print("\n══════ Comparing strategies ══════")
        results = agent.compare_strategies(
            labeled_df=labeled_df,
            pool_df=pool_df,
            n_iterations=args.iterations,
            batch_size=args.batch_size,
            output_path=str(out / "learning_curve.png"),
        )
        for strat, hist in results.items():
            print(f"\n  Strategy: {strat}")
            for entry in hist:
                print(
                    f"    iter {entry['iteration']}: "
                    f"n={entry['n_labeled']}, "
                    f"acc={entry['accuracy']:.4f}, "
                    f"f1={entry['f1']:.4f}"
                )
    else:
        print(f"\n══════ AL Cycle ({args.strategy}) ══════")
        history = agent.run_cycle(
            labeled_df=labeled_df,
            pool_df=pool_df,
            strategy=args.strategy,
            n_iterations=args.iterations,
            batch_size=args.batch_size,
        )
        for entry in history:
            print(
                f"  iter {entry['iteration']}: "
                f"n={entry['n_labeled']}, "
                f"acc={entry['accuracy']:.4f}, "
                f"f1={entry['f1']:.4f}"
            )
        agent.report(history, output_path=str(out / "learning_curve.png"))

    print(f"\nPlot saved → {out / 'learning_curve.png'}")


if __name__ == "__main__":
    _cli()
