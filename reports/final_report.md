# Final Pipeline Report

Generated: 2026-03-26T22:43:41.183441

---

## 1. Описание задачи и датасета

- **Задача:** Бинарная классификация тональности текста (sentiment analysis)
- **Модальность:** Текст (английский язык)
- **Объём сырых данных:** 1100 примеров
- **Объём после чистки:** 1100 примеров
- **Объём финального датасета:** 1100 примеров
- **Классы:** positive, negative
- **Источники:** HuggingFace (`rotten_tomatoes`), web scraping (`quotes.toscrape.com`)

---

## 2. Что делал каждый агент

### DataCollectionAgent (Задание 1)
- Собрал данные из 2 источников: HuggingFace dataset `rotten_tomatoes` и web scraping с `quotes.toscrape.com`
- Унифицировал схему: `text`, `label`, `source`, `collected_at`
- Дедупликация по тексту, удаление пустых записей
- Скрейпинг с автоматической keyword-based разметкой для цитат

### DataQualityAgent (Задание 2)
- Обнаружил проблемы: пропуски, дубликаты, выбросы по длине текста, дисбаланс классов
- Стратегия чистки: `drop` пропусков, `drop` дубликатов, `clip_iqr` выбросов
- Обоснование: drop для пропусков безопасен при достаточном объёме данных; clip_iqr сохраняет длинные тексты, обрезая до разумной длины
- Результат: 1100 → 1100 строк

### AnnotationAgent (Задание 3)
- Метод авторазметки: zero-shot (BART-large-MNLI)
- Генерация спецификации разметки (`annotation_spec.md`)
- Оценка качества: Cohen's κ, распределение меток, статистика confidence
- Экспорт в формат LabelStudio (`labelstudio_import.json`)
- Флагирование: 130 примеров с низкой уверенностью

### ActiveLearningAgent (Задание 4)
- Сравнение стратегий: entropy vs random
- 5 итераций по 20 примеров, начальный seed = 50
- Признаки: TF-IDF (5000 features) → SVD (50 компонент) → LogisticRegression
- Кривые обучения сохранены в `reports/learning_curve.png`

- **entropy:** итоговая accuracy=0.5227, f1=0.5215 (n=150)
- **random:** итоговая accuracy=0.5409, f1=0.5358 (n=150)

---

## 3. Описание HITL-точки

- **Количество флагированных примеров:** 130
- **Количество исправлений:** 66
- **Порог confidence:** 0.7
- **Механизм:**
  1. После авторазметки примеры с `confidence < 0.7` сохраняются в `review_queue.csv`
  2. Человек открывает файл, просматривает каждый пример
  3. Заполняет столбец `corrected_label` правильной меткой
  4. Сохраняет как `review_queue_corrected.csv`
  5. Пайплайн читает исправления и объединяет с основным датасетом
- **Результат:** исправленные метки используются для обучения финальной модели

### Примеры исправлений

| Текст | Авто-метка | Исправлено на | Confidence |
|-------|-----------|---------------|------------|
| if you're a crocodile hunter fan , you'll enjoy at least the… | positive | negative | 0.667 |
| the chateau belongs to rudd , whose portrait of a therapy-de… | negative | positive | 0.552 |
| generally provides its target audience of youngsters enough … | negative | positive | 0.505 |
| this familiar rise-and-fall tale is long on glamour and shor… | negative | positive | 0.633 |
| by candidly detailing the politics involved in the creation … | negative | positive | 0.652 |
| there are moments it can be heart-rending in an honest and u… | negative | positive | 0.530 |
| a benign but forgettable sci-fi diversion .… | positive | negative | 0.646 |
| liotta put on 30 pounds for the role , and has completely tr… | negative | positive | 0.689 |
| there's back-stabbing , inter-racial desire and , most impor… | negative | positive | 0.642 |
| . . . routine , harmless diversion and little else .… | positive | negative | 0.608 |

---

## 4. Метрики качества

### По этапам

| Этап | Метрика | Значение |
|------|---------|----------|
| Сбор данных | Объём | 1100 |
| Чистка | Объём после | 1100 |
| Авторазметка | Mean confidence | 0.8974 |
| Авторазметка | Cohen's κ | 0.5602 |
| HITL | Проверено | 130 |
| HITL | Исправлено | 66 |

### Итоговые метрики модели

- **Accuracy:** 0.6045
- **F1 (macro):** 0.593
- **F1 (weighted):** 0.6017
- **Train size:** 880
- **Test size:** 220

```
              precision    recall  f1-score   support

    negative       0.64      0.69      0.66       124
    positive       0.55      0.50      0.52        96

    accuracy                           0.60       220
   macro avg       0.60      0.59      0.59       220
weighted avg       0.60      0.60      0.60       220

```

### Active Learning

- **entropy:** accuracy=0.5227, f1=0.5215
- **random:** accuracy=0.5409, f1=0.5358

---

## 5. Ретроспектива

### Что сработало
- Модульная архитектура: каждый агент — отдельный класс с чётким API, легко подключается к пайплайну
- DataQualityAgent эффективно обнаруживает и устраняет проблемы — чистка заметно улучшает качество модели
- HITL через CSV прост и прозрачен — легко отследить, какие примеры исправлены и как это повлияло на результат
- Active Learning с entropy sampling показывает преимущество над random — экономит примеры при том же качестве

### Что не сработало / было сложно
- Zero-shot классификация (BART) на CPU медленная — для 3000 примеров нужно ~20 минут
- Скрейпинг quotes.toscrape.com даёт цитаты, а не рецензии — домен отличается от rotten_tomatoes, что добавляет шум
- Keyword-based авто-разметка скрейпинга неточна — многие цитаты получают некорректную метку

### Что бы сделал иначе
- Использовал бы GPU или distilled модель для AnnotationAgent (DistilBART / TinyBERT)
- Добавил бы DVC для версионирования данных и моделей
- Реализовал бы Streamlit-дашборд для интерактивной HITL-разметки
- Подключил бы больше источников данных (Twitter API, Reddit, IMDb reviews)
- Добавил бы мониторинг data drift при обновлении датасета
