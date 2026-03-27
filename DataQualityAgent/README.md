# DataQualityAgent — «Детектив данных»

Агент автоматически выявляет и устраняет проблемы качества данных.

## Установка

```bash
pip install -r requirements.txt
```

## Быстрый старт

```python
from agents.data_quality_agent import DataQualityAgent
import pandas as pd

df = pd.read_csv("data.csv")
agent = DataQualityAgent()

# 1. Обнаружение проблем
report = agent.detect_issues(df)
# → {'missing': {...}, 'duplicates': N, 'outliers': [...], 'imbalance': {...}}

# 2. Чистка
df_clean = agent.fix(df, strategy={
    'missing': 'median',
    'duplicates': 'drop',
    'outliers': 'clip_iqr'
})

# 3. Сравнение до/после
comparison = agent.compare(df, df_clean)
```

## API

### `detect_issues(df) → dict`

Обнаруживает 4 типа проблем:
- **missing** — пропущенные значения (NaN + пустые строки)
- **duplicates** — полные дубликаты строк
- **outliers** — выбросы по IQR/z-score (числовые столбцы + длина текста)
- **imbalance** — дисбаланс классов в целевом столбце

### `fix(df, strategy) → DataFrame`

Стратегии чистки:

| Проблема | Стратегии |
|---|---|
| missing | `drop`, `median`, `mode`, `ffill`, `constant` |
| duplicates | `drop` |
| outliers | `clip_iqr`, `clip_zscore`, `drop_iqr`, `drop_zscore` |
| imbalance | `oversample`, `undersample` |

### `compare(df_before, df_after) → DataFrame`

Таблица «было / стало» по каждой метрике качества.

## CLI

```bash
python agents/data_quality_agent.py data.csv --strategy "missing=median,duplicates=drop,outliers=clip_iqr" -o clean.csv
```

## Ноутбук

Полный анализ с визуализациями: `notebooks/data_quality.ipynb`

## Структура

```
DataQualityAgent/
├── agents/
│   ├── __init__.py
│   └── data_quality_agent.py
├── notebooks/
│   └── data_quality.ipynb
├── data/                          # создаётся при запуске ноутбука
│   ├── sentiment_clean.csv
│   └── comparison_report.csv
├── requirements.txt
└── README.md
```
