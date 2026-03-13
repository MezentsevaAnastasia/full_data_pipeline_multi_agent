# Text2SQL CLI Agent

## Что это

В проекте создан отдельный терминальный агент, который можно запускать без Cursor skills:

- входная точка: `run_text2sql_agent.py`
- пакет агента: `text2sql_agent/`

Агент:

- загружает открытые `text2sql` benchmark-датасеты
- нормализует записи в единую схему
- сохраняет сырые данные
- собирает итоговый CSV
- делает `README.md` с кратким отчетом
- при желании обогащает записи через бесплатный `Gemini`

## Структура

```text
run_text2sql_agent.py
text2sql_agent/
├── cli.py
├── enrichment.py
├── models.py
├── reporting.py
├── requirements.txt
└── sources.py
```

## Установка зависимостей

```bash
pip install -r text2sql_agent/requirements.txt
```

## Базовый запуск

```bash
python run_text2sql_agent.py
```

По умолчанию агент:

- берет источники `spider,wikisql`
- пытается подобрать рабочий split автоматически
- сохраняет результат в `data/text2sql_dataset.csv`
- сохраняет raw-файлы в `data/raw/`
- создает отчет `data/README.md`

## Полезные команды

Собрать 100 записей:

```bash
python run_text2sql_agent.py --max-records 100
```

Использовать только `spider`:

```bash
python run_text2sql_agent.py --sources spider --max-records 150
```

Фильтровать по доменной подстроке:

```bash
python run_text2sql_agent.py --domain-filter student
```

Сохранить в другую папку:

```bash
python run_text2sql_agent.py --output-dir output_text2sql
```

## Обогащение через бесплатную LLM

Агент поддерживает опциональное обогащение через `gemini-2.0-flash`.

### 1. Получить API key

Создайте ключ в Google AI Studio.

### 2. Экспортировать переменную окружения

```bash
export GEMINI_API_KEY="your_key"
```

### 3. Запустить enrichment

```bash
python run_text2sql_agent.py --enrich
```

По умолчанию будут добавлены:

- `difficulty`
- `sql_features`

Можно явно задать задачи:

```bash
python run_text2sql_agent.py --enrich --enrich-tasks difficulty,sql_features,domain
```

## Какие поля будут в датасете

```text
question, sql, schema, db_id, db_dialect, domain, difficulty, sql_features, source_url, source_name, collected_at
```

## Что создаст агент

После запуска появятся:

- `data/raw/spider.jsonl`
- `data/raw/wikisql.jsonl`
- `data/text2sql_dataset.csv`
- `data/README.md`

При `--enrich` будет обновлен итоговый CSV с derived fields.

## Пример для задания

Если нужно просто показать, что агент работает и запускается из терминала:

```bash
python run_text2sql_agent.py --sources spider,wikisql --max-records 50
```

## Ограничения

- Сейчас агент ориентирован на источники `spider` и `wikisql`
- Для загрузки нужен интернет
- Для enrichment нужен `GEMINI_API_KEY`
- Для `wikisql` SQL может быть собран из структурированного представления, а не взят как исходная строка
