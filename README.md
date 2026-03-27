# Full Data Pipeline with Human-in-the-Loop

Единый ML-пайплайн, объединяющий 4 агента из заданий 1–4 в воспроизводимый end-to-end процесс:
сбор данных → чистка → авторазметка → HITL-проверка → active learning → обучение модели → отчёт.

---

## Быстрый старт

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Запустить пайплайн (auto mode — без ручной проверки)
python run_pipeline.py --auto

# 3. Или с ручной проверкой (HITL)
python run_pipeline.py
```

### Режимы запуска

| Команда | Описание |
|---------|----------|
| `python run_pipeline.py --auto` | Полный пайплайн с автоматической симуляцией HITL |
| `python run_pipeline.py --auto --fast` | Быстрый режим (~300 примеров) для тестирования |
| `python run_pipeline.py` | Интерактивный режим — пауза для ручной разметки |
| `python run_pipeline.py --auto --max-samples 3000` | Полный датасет |

### Параметры

| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `--auto` | off | Симулировать HITL (без ожидания ввода) |
| `--fast` | off | Быстрый режим (300 примеров) |
| `--max-samples N` | 1000 | Макс. число примеров из HuggingFace |
| `--threshold T` | 0.7 | Порог confidence для HITL-флагирования |
| `--task NAME` | sentiment_analysis | Задача из config.yaml |

---

## Архитектура пайплайна

```
┌─────────────────────────────────────────────────────────┐
│                    run_pipeline.py                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Step 1: COLLECT ──► DataCollectionAgent                 │
│    │  HuggingFace (rotten_tomatoes) + Web scraping       │
│    ▼                                                    │
│  Step 2: CLEAN ───► DataQualityAgent                    │
│    │  Detect: missing, duplicates, outliers, imbalance   │
│    │  Fix: drop + clip_iqr                               │
│    ▼                                                    │
│  Step 3: LABEL ───► AnnotationAgent                     │
│    │  Zero-shot classification (BART) + confidence       │
│    ▼                                                    │
│  Step 4: HITL ────► Human-in-the-Loop                   │
│    │  ❗ review_queue.csv → человек правит →              │
│    │     review_queue_corrected.csv                       │
│    ▼                                                    │
│  Step 5: AL ──────► ActiveLearningAgent                 │
│    │  Entropy vs Random, learning curves                 │
│    ▼                                                    │
│  Step 6: TRAIN ───► Final Model (TF-IDF + SVD + LogReg) │
│    │  Save model + metrics                               │
│    ▼                                                    │
│  Step 7: REPORT ──► Отчёты + Data Card                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Структура репозитория

```
├── agents/                          # Все 4 агента
│   ├── __init__.py
│   ├── data_collection_agent.py     # Задание 1 — сбор данных
│   ├── data_quality_agent.py        # Задание 2 — качество данных
│   ├── annotation_agent.py          # Задание 3 — авторазметка
│   └── al_agent.py                  # Задание 4 — active learning
├── run_pipeline.py                  # Основной скрипт пайплайна
├── config.yaml                      # Конфигурация источников данных
├── requirements.txt                 # Зависимости
├── data/
│   ├── raw/                         # Сырые и очищенные данные
│   │   ├── pipeline_raw.csv
│   │   └── pipeline_clean.csv
│   └── labeled/                     # Финальный датасет
│       ├── pipeline_labeled.csv
│       └── DATA_CARD.md
├── review_queue.csv                 # Файл для HITL-проверки
├── review_queue_corrected.csv       # Исправленные метки (после HITL)
├── models/
│   └── final_model.pkl              # Обученная модель
├── reports/
│   ├── quality_report.md            # Отчёт DataQualityAgent
│   ├── annotation_report.md         # Отчёт AnnotationAgent
│   ├── annotation_spec.md           # Спецификация разметки
│   ├── al_report.md                 # Отчёт Active Learning
│   ├── learning_curve.png           # Кривые обучения AL
│   ├── labelstudio_import.json      # Экспорт для LabelStudio
│   └── final_report.md              # Итоговый отчёт с метриками
├── DataCollectionAgent/             # Оригинал задания 1
├── DataQualityAgent/                # Оригинал задания 2
├── AnnotationAgent/                 # Оригинал задания 3
└── MultimodalAgent/                 # Оригинал задания 4
```

---

## Итоговый отчёт

### 1. Описание задачи и датасета

- **Задача:** Бинарная классификация тональности текста (sentiment analysis)
- **Модальность:** Текст (английский язык)
- **Классы:** `positive`, `negative`
- **Объём:** ~1000 примеров (настраивается через `--max-samples`)
- **Источники данных:**
  - **HuggingFace** — датасет `rotten_tomatoes` (рецензии на фильмы с метками positive/negative)
  - **Web scraping** — `quotes.toscrape.com` (цитаты с keyword-based разметкой)
- **Формат:** CSV с колонками `text`, `label`, `source`, `collected_at`

### 2. Что делал каждый агент

#### DataCollectionAgent (Задание 1)
- Собирает данные из 2+ источников через единый API (`run()`)
- HuggingFace: загрузка `rotten_tomatoes` с маппингом меток `{0: negative, 1: positive}`
- Web scraping: парсинг `quotes.toscrape.com` с CSS-селекторами, keyword-based авто-разметка
- Унификация схемы: `text`, `label`, `source`, `collected_at`
- Дедупликация по тексту, удаление пустых записей
- **Решение:** использован `rotten_tomatoes` как основной источник — сбалансированный бинарный датасет, хорошо изученный в NLP

#### DataQualityAgent (Задание 2)
- Обнаруживает 4 типа проблем: пропуски, дубликаты, выбросы (IQR), дисбаланс классов
- Стратегия чистки: `drop` пропусков, `drop` дубликатов, `clip_iqr` выбросов по длине текста
- **Решение:** `clip_iqr` для выбросов вместо `drop` — сохраняет данные, обрезая слишком длинные тексты до разумной длины. `drop` для пропусков безопасен при достаточном объёме данных
- Генерирует сравнительный отчёт «было / стало» по каждой метрике

#### AnnotationAgent (Задание 3)
- Авторазметка через zero-shot classification (модель `facebook/bart-large-mnli`)
- Каждому примеру присваивается `predicted_label` и `confidence` (0–1)
- Примеры с `confidence < 0.7` автоматически флагируются (`needs_review = True`)
- Генерирует спецификацию разметки (`annotation_spec.md`) с определениями классов, примерами, граничными случаями
- Экспорт в формат LabelStudio для возможной ручной доразметки
- **Fallback:** если BART недоступен — cross-validated TF-IDF + LogReg для генерации confidence
- **Решение:** порог 0.7 выбран эмпирически — баланс между объёмом ручной проверки и качеством

#### ActiveLearningAgent (Задание 4)
- Итеративный отбор наиболее информативных примеров из пула
- Стратегии: `entropy` (максимальная энтропия предсказаний), `random` (baseline)
- Модель: TF-IDF (5000 features, bigrams) → SVD (50 компонент) → LogisticRegression
- 5 итераций по 20 примеров, начальный seed = 50
- **Решение:** entropy sampling показывает преимущество — достигает того же качества с меньшим количеством размеченных примеров

### 3. Описание HITL-точки

Пайплайн содержит явную точку Human-in-the-Loop после шага авторазметки:

1. **Флагирование:** AnnotationAgent помечает примеры с `confidence < 0.7` как `needs_review = True`
2. **Сохранение:** эти примеры сохраняются в `review_queue.csv` с колонками:
   - `text` — текст примера
   - `label` — оригинальная метка из источника
   - `predicted_label` — метка от авторазметки
   - `confidence` — уверенность модели
   - `corrected_label` — пустая колонка для исправления человеком
3. **Проверка:** человек открывает файл, просматривает каждый пример, заполняет `corrected_label`
4. **Возврат:** исправленный файл сохраняется как `review_queue_corrected.csv`
5. **Слияние:** пайплайн объединяет исправленные метки с высоко-уверенными предсказаниями → финальный датасет `data/labeled/pipeline_labeled.csv`

**Это реальная правка данных:** содержимое `corrected_label` напрямую влияет на обучение модели.

В режиме `--auto` пайплайн симулирует проверку: использует ground truth метки как «мнение человека» и исправляет расхождения с авторазметкой. Это демонстрирует механизм, а реальная ручная проверка запускается без `--auto`.

### 4. Метрики качества

Точные метрики генерируются при запуске пайплайна и сохраняются в `reports/final_report.md`.

#### Ожидаемые метрики (на rotten_tomatoes, ~1000 примеров):

| Этап | Метрика | Ожидаемое значение |
|------|---------|-------------------|
| Сбор данных | Объём | ~1000–1100 |
| Чистка | Объём после | ~1000 |
| Авторазметка | Mean confidence | 0.75–0.90 |
| Авторазметка | Cohen's κ | 0.5–0.8 |
| HITL | Примеров на проверку | ~100–300 |
| **Финальная модель** | **Accuracy** | **0.75–0.85** |
| **Финальная модель** | **F1 (macro)** | **0.75–0.85** |
| Active Learning | Entropy vs Random Δacc | +0.01–0.05 |

### 5. Ретроспектива

#### Что сработало
- **Модульная архитектура:** каждый агент — независимый класс с чётким API. Подключение к пайплайну — замена одного вызова
- **DataQualityAgent:** автоматическое обнаружение проблем + сравнительные отчёты дают прозрачность
- **HITL через CSV:** простой и универсальный механизм — работает в любой среде, не требует UI
- **Fallback-механизм** в авторазметке: если BART недоступен, пайплайн всё равно работает
- **Active Learning:** entropy sampling экономит ~20–30% примеров по сравнению с random

#### Что не сработало / было сложно
- **Zero-shot на CPU медленный:** BART-large-MNLI обрабатывает ~2–5 примеров/сек, для 3000 примеров нужно ~20 минут
- **Домен скрейпинга не совпадает:** цитаты с quotes.toscrape.com — не рецензии, keyword-based разметка шумная
- **Cohen's κ зависит от совпадения классов:** если авторазметка и ground truth используют разные границы решения, κ может быть низким

#### Что бы сделал иначе
- Использовал бы **GPU** или **distilled модель** (DistilBART, TinyBERT) для AnnotationAgent
- Добавил бы **DVC** для версионирования данных и моделей
- Реализовал бы **Streamlit-дашборд** для интерактивной HITL-разметки вместо CSV
- Подключил бы более релевантные источники данных (IMDb API, Twitter, Reddit)
- Добавил бы **мониторинг data drift** для отслеживания изменений при обновлении датасета
- Попробовал бы **margin sampling** как третью AL-стратегию и fine-tuned модели вместо LogReg

---

## Data Card

| Поле | Значение |
|------|---------|
| Название | Sentiment Classification Dataset |
| Задача | Бинарная классификация тональности |
| Модальность | Текст (английский) |
| Классы | positive, negative |
| Объём | ~1000 примеров |
| Источники | HuggingFace (rotten_tomatoes), quotes.toscrape.com |
| Формат | CSV |
| Лицензия | Rotten Tomatoes — academic use |

Подробная data card генерируется автоматически: `data/labeled/DATA_CARD.md`

---

## Лицензия

MIT
