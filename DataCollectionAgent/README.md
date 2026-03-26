# DataCollectionAgent — Text Sentiment Analysis

## Задача ML

**Бинарная классификация тональности текста** (positive / negative).

Агент собирает текстовые данные из двух источников, приводит к единой схеме и сохраняет унифицированный датасет, готовый для обучения модели сентимент-анализа.

### Источники данных

| # | Тип | Источник | Описание |
|---|-----|----------|----------|
| 1 | HuggingFace dataset | `rotten_tomatoes` | Короткие рецензии на фильмы (pos/neg) |
| 2 | Web scraping | `quotes.toscrape.com` | Цитаты с автоматической разметкой тональности |

## Схема данных

| Колонка | Тип | Описание |
|---------|-----|----------|
| `text` | str | Текст рецензии или цитаты |
| `label` | str | Метка тональности: `positive` / `negative` |
| `source` | str | Идентификатор источника |
| `collected_at` | str | Временная метка сбора (ISO 8601, UTC) |

## Установка

```bash
pip install -r requirements.txt
```

## Запуск

### Показать доступные задачи

```bash
python agents/data_collection_agent.py --list
```

```
Доступные задачи:

  • sentiment_analysis             Бинарная классификация тональности текста (positive / negative)
  • topic_classification           Классификация новостей по темам (World / Sports / Business / Sci-Tech)
  • emotion_detection              Определение эмоции текста (sadness / joy / love / anger / fear / surprise)
```

### Произвольная тема (natural language)

Агент сам найдёт подходящий датасет на HuggingFace Hub, определит поля и метки:

```bash
python agents/data_collection_agent.py --topic "toxic comment classification"
python agents/data_collection_agent.py --topic "spam detection in emails"
python agents/data_collection_agent.py --topic "emotion detection in tweets"
python agents/data_collection_agent.py --topic "fake news detection"
```

Результат сохраняется в `data/raw/<topic_slug>_dataset.csv`.

### Запуск по конкретной задаче из конфига

```bash
python agents/data_collection_agent.py --task sentiment_analysis
python agents/data_collection_agent.py --task topic_classification
python agents/data_collection_agent.py --task emotion_detection
```

### Python API

```python
from agents.data_collection_agent import DataCollectionAgent

agent = DataCollectionAgent(config='config.yaml')

# Произвольная тема — автопоиск датасета
df = agent.run(topic='toxic comment classification')

# Готовая задача из config.yaml
df = agent.run(task='sentiment_analysis')

# Полностью кастомные источники
df = agent.run(sources=[
    {'type': 'hf_dataset', 'name': 'imdb',
     'label_map': {0: 'negative', 1: 'positive'}, 'max_samples': 1000},
    {'type': 'scrape', 'url': 'https://quotes.toscrape.com',
     'selector': 'div.quote span.text', 'auto_label': True, 'max_pages': 3},
])
```

## Конфигурация

**`--topic`** — агент ищет на HuggingFace Hub датасет по теме, автоматически определяет текстовое поле, поле меток и label_map. Если точный запрос не дал результатов, запрос упрощается.

**`--task`** — готовые задачи из `config.yaml` (секция `tasks`).

Поддерживаемые типы источников:

- `hf_dataset` — датасет с HuggingFace Hub (`name`, `split`, `label_map`, `max_samples`)
- `scrape` — веб-скрейпинг (`url`, `selector`, `max_pages`, `auto_label`)
- `api` — REST API (`endpoint`, `params`, `results_key`, `auto_label`)

Новые готовые задачи добавляются блоком в `tasks` в `config.yaml`.

## EDA

Ноутбук `notebooks/eda.ipynb` содержит:

- Распределение классов (bar chart)
- Распределение длин текстов (histogram)
- Топ-20 самых частых слов (bar chart)
- Облако слов (word cloud)
- Распределение по источникам (pie chart)

## Структура проекта

```
DataCollectionAgent/
├── agents/
│   ├── __init__.py
│   └── data_collection_agent.py   # Агент
├── config.yaml                    # Конфигурация источников
├── notebooks/
│   └── eda.ipynb                  # EDA и визуализации
├── data/
│   └── raw/                       # Собранные данные
├── requirements.txt
├── README.md
└── task_1.md                      # Описание задания
```
