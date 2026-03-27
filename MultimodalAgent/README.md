# ActiveLearningAgent (Задание 4, Трек A)

Агент для умного отбора данных (Active Learning) — итеративно выбирает наиболее информативные примеры из пула неразмеченных данных для обучения модели.

## Архитектура

```
ActiveLearningAgent
├── fit(labeled_df)                 → обученная модель (TF-IDF + классификатор)
├── query(pool_df, strategy)        → индексы наиболее информативных примеров
├── evaluate(labeled_df, test_df)   → метрики (accuracy, F1)
├── report(history)                 → learning_curve.png
├── run_cycle(...)                  → история AL-цикла
└── compare_strategies(...)         → сравнение нескольких стратегий
```

## Стратегии отбора

| Стратегия | Описание |
|-----------|----------|
| `entropy` | Отбирает примеры с наибольшей энтропией предсказания (максимальная неопределённость) |
| `margin`  | Отбирает примеры с наименьшим отрывом между двумя наиболее вероятными классами |
| `random`  | Случайный отбор (baseline) |

## Быстрый старт

```bash
pip install -r requirements.txt

# Запуск AL-цикла (entropy)
python -m agents.al_agent ../DataCollectionAgent/data/raw/dataset.csv \
    --strategy entropy --seed-size 50 --iterations 5 --batch-size 20

# Сравнение entropy vs random
python -m agents.al_agent ../DataCollectionAgent/data/raw/dataset.csv \
    --compare --seed-size 50 --iterations 5 --batch-size 20
```

## Использование в коде

```python
from agents.al_agent import ActiveLearningAgent

agent = ActiveLearningAgent(model='logreg')

# Цикл: старт с N=50, 5 итераций по 20 примеров
history = agent.run_cycle(
    labeled_df=df_labeled_50,
    pool_df=df_unlabeled,
    test_df=df_test,
    strategy='entropy',
    n_iterations=5,
    batch_size=20,
)
# → history: список {iteration, n_labeled, accuracy, f1, strategy}

agent.report(history)  # → learning_curve.png
```

### Сравнение стратегий

```python
results = agent.compare_strategies(
    labeled_df=df_seed,
    pool_df=df_pool,
    test_df=df_test,
    strategies=['entropy', 'margin', 'random'],
    n_iterations=10,
    batch_size=20,
    output_path='learning_curve.png',
)
```

## Структура

```
MultimodalAgent/
├── agents/
│   ├── __init__.py
│   └── al_agent.py          # ActiveLearningAgent
├── notebooks/
│   └── al_experiment.ipynb   # Эксперимент: entropy vs margin vs random
├── requirements.txt
├── README.md
└── task4.md                  # Описание задания
```

## Результаты эксперимента

Эксперимент проведён на датасете Rotten Tomatoes (sentiment classification, ~3100 примеров).

- **Seed:** 50 размеченных примеров
- **Pool:** ~2550 примеров
- **Test:** 500 примеров
- **Итерации:** 5 по 20 примеров

### Выводы

1. Стратегии `entropy` и `margin` обеспечивают более быстрый рост качества модели по сравнению с `random` baseline
2. При использовании entropy-стратегии модель достигает того же уровня accuracy/F1 с меньшим количеством размеченных примеров
3. Рекомендация: для задачи sentiment classification — стратегия `entropy`

## Зависимости

- pandas, numpy — работа с данными
- scikit-learn — TF-IDF, LogisticRegression, метрики
- matplotlib — визуализация кривых обучения
