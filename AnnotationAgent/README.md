# AnnotationAgent

Агент для автоматической разметки данных, генерации спецификации, оценки качества и экспорта в LabelStudio.

## Установка

```bash
pip install -r requirements.txt
```

## Быстрый старт

```python
from agents import AnnotationAgent
import pandas as pd

df = pd.read_csv("../DataCollectionAgent/data/raw/sentiment_dataset.csv")

agent = AnnotationAgent(modality="text")

# 1. Автоматическая разметка (zero-shot classification)
df_labeled = agent.auto_label(df)

# 2. Генерация спецификации разметки
agent.generate_spec(df_labeled, task="sentiment_classification")
# → annotation_spec.md

# 3. Оценка качества
metrics = agent.check_quality(df_labeled)
# → {'kappa': 0.72, 'label_dist': {...}, 'confidence_mean': 0.85, ...}

# 4. Экспорт в LabelStudio
agent.export_to_labelstudio(df_labeled)
# → labelstudio_import.json

# 5. HITL: флагирование для ручной проверки
review_df = agent.flag_for_review(df_labeled)
# → review_queue.csv
```

## CLI

```bash
python -m agents.annotation_agent data.csv \
    --modality text \
    --task sentiment_classification \
    --threshold 0.7 \
    --output-dir output/
```

## Skills

| Skill | Описание | Вход | Выход |
|-------|----------|------|-------|
| `auto_label(df)` | Zero-shot классификация текста | DataFrame | DataFrame + predicted_label, confidence |
| `generate_spec(df, task)` | Генерация спецификации разметки | DataFrame + название задачи | Markdown-файл |
| `check_quality(df_labeled)` | Оценка качества (κ, распределение, confidence) | DataFrame с метками | dict с метриками |
| `export_to_labelstudio(df)` | Экспорт в формат LabelStudio | DataFrame | JSON-файл |
| `flag_for_review(df_labeled)` | HITL: выборка для ручной разметки | DataFrame с confidence | CSV-файл |

## Модальности

Реализована поддержка модальности **text** (zero-shot classification через `facebook/bart-large-mnli`).

Архитектура предусматривает расширение на audio (Whisper) и image (CLIP/YOLO).

## Структура

```
AnnotationAgent/
├── agents/
│   ├── __init__.py
│   └── annotation_agent.py
├── notebooks/
│   └── annotation.ipynb
├── requirements.txt
├── README.md
└── task3.md
```
