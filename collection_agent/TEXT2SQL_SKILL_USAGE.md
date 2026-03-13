# Text2SQL Data Collector

## Что это

В проекте есть скилл `.cursor/skills/data-collector/`, адаптированный под сбор датасетов для `text2sql`.

Он помогает:

- находить открытые `text2sql` датасеты
- собирать пары `question -> sql`
- сохранять схему базы
- очищать и объединять данные
- при желании обогащать записи через бесплатную LLM

## Готовый промпт для агента

Скопируйте этот запрос в чат Cursor внутри проекта:

```text
Используй скилл data-collector.

Собери датасет для text2sql задачи.

Требования:
- нужны пары natural language question -> SQL query
- добавь schema или краткое описание таблиц
- формат результата: CSV
- целевой размер: 100-300 записей
- приоритетные источники: Spider, WikiSQL, BIRD, SParC, CoSQL, Hugging Face, GitHub-репозитории с открытыми benchmark-файлами

Обязательные поля:
- question
- sql
- schema
- db_id
- db_dialect
- domain
- difficulty
- sql_features
- source_url
- source_name
- collected_at

Правила:
- сохраняй сырые файлы в data/raw/
- итоговый файл сохрани в data/text2sql_dataset.csv
- если получится, сделай enriched-версию в data/text2sql_dataset_enriched.csv
- создай отчет data/README.md
- не придумывай синтетические записи без явного разрешения
- если difficulty или sql_features отсутствуют в источнике, можешь вывести их как derived fields
```

## Короткий вариант промпта

```text
Используй скилл data-collector и собери text2sql датасет в CSV из открытых источников. Нужны поля: question, sql, schema, db_id, db_dialect, domain, difficulty, sql_features, source_url, source_name, collected_at. Сохрани raw в data/raw, итог в data/text2sql_dataset.csv и сделай data/README.md.
```

## Как запустить в Cursor

1. Открой этот проект в Cursor.
2. Убедись, что скилл лежит в `.cursor/skills/data-collector/`.
3. Открой чат агента.
4. Вставь готовый промпт из раздела выше.
5. Дождись, пока агент:
   - найдет источники
   - скачает или соберет сырые данные
   - сохранит файлы в `data/raw/`
   - соберет финальный CSV
   - создаст `data/README.md`

## Ручной запуск скриптов

Если сырые файлы уже есть, можно собрать датасет вручную.

### 1. Установить зависимости

```bash
pip install -r .cursor/skills/data-collector/scripts/requirements.txt
```

### 2. Подготовить структуру папок

```bash
mkdir -p data/raw
```

### 3. Положить сырые файлы в `data/raw/`

Поддерживаемые форматы:

- `.csv`
- `.json`
- `.jsonl`
- `.xlsx`
- `.parquet`

### 4. Собрать итоговый датасет

```bash
python .cursor/skills/data-collector/scripts/save_data.py --input data/raw --output data/text2sql_dataset.csv --format csv
```

### 5. Опционально обогатить через бесплатную LLM

Получите ключ в Google AI Studio и экспортируйте его:

```bash
export GEMINI_API_KEY="your_key"
```

Примеры:

Добавить теги по вопросу:

```bash
python .cursor/skills/data-collector/scripts/llm_enrich.py --input data/text2sql_dataset.csv --output data/text2sql_dataset_enriched.csv --text-column question --task tags
```

Оценить сложность:

```bash
python .cursor/skills/data-collector/scripts/llm_enrich.py --input data/text2sql_dataset.csv --output data/text2sql_dataset_enriched.csv --text-column sql --task difficulty
```

Извлечь SQL features:

```bash
python .cursor/skills/data-collector/scripts/llm_enrich.py --input data/text2sql_dataset.csv --output data/text2sql_dataset_enriched.csv --text-column sql --task sql_features
```

## Что должно получиться

Минимально ожидаемый результат:

- `data/raw/` с исходными файлами
- `data/text2sql_dataset.csv`
- `data/README.md`

Опционально:

- `data/text2sql_dataset_enriched.csv`

## Рекомендуемые источники

- Spider
- WikiSQL
- BIRD
- SParC
- CoSQL
- Hugging Face Datasets
- GitHub-репозитории с открытыми benchmark JSON/CSV

## Рекомендуемая схема

```text
question, sql, schema, db_id, db_dialect, domain, difficulty, sql_features, source_url, source_name, collected_at
```
