Data Project — Финальный пайплайн | 40 баллов | Дедлайн: занятие 10 (28.03.2026)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ЦЕЛЬ

Собрать всех агентов из заданий 1–4 в единый дата-пайплайн с human-in-the-loop. Инструмент оркестрации — на ваш выбор. Главное: пайплайн должен быть единым, воспроизводимым и содержать точки, в которых человек проверяет или правит работу агентов. На выходе — готовый размеченный датасет, обученная модель, итоговый отчёт. Проект — ваше ML-портфолио.

ЛОГИКА ПАЙПЛАЙНА

- Шаг 1: Сбор — DataCollectionAgent собирает данные из 2+ источников
- Шаг 2: Чистка — DataQualityAgent выявляет и устраняет проблемы
- Шаг 3: Авторазметка — AnnotationAgent размечает данные автоматически
- ❗ Human-in-the-loop: агент флагает неуверенные примеры → человек проверяет и правит метки
- Шаг 4: Отбор — ALAgent выбирает наиболее информативные примеры для разметки
- Шаг 5: Обучение — модель обучается на размеченных данных
- Шаг 6: Отчёт — метрики на каждом этапе + ретроспектива

ЧЕМ МОЖНО ОРКЕСТРОВАТЬ ПАЙПЛАЙН

Инструмент оркестрации — на выбор студента. Важно не чем, а то, что пайплайн единый, воспроизводимый и запускается одной командой.

- Python-скрипт (run_pipeline.py) — самый простой вариант: вызов агентов последовательно, сохранение промежуточных результатов в parquet/csv
- Prefect — лёгкий Python-оркестратор, @flow и @task декораторы, визуальный UI, хорошо подходит для новичков
- Airflow — популярный пром инструмент для ML-пайплайнов, DAG-архитектура, сложнее в настройке
- Dagster
- Metaflow (Netflix) — удобен для ML, поддержка артефактов и версионирования данных
- Jupyter Notebook (линейный) — приемлемо, если каждый этап оформлен отдельной ячейкой + сохраняет промежуточные файлы

ПРИМЕР НА Prefect (рекомендуем новичкам)

from prefect import flow, task
from agents.data_collection_agent import DataCollectionAgent
from agents.data_quality_agent import DataQualityAgent
from agents.annotation_agent import AnnotationAgent
from agents.al_agent import ActiveLearningAgent

@task
def collect(): return DataCollectionAgent().run(...)

@task
def clean(df): return DataQualityAgent().fix(df, strategy={...})

@task
def auto_label(df): return AnnotationAgent(modality='text').auto_label(df)

@task
def human_review(df):
    # ❗ Человек проверяет флаги confidence < 0.7
    low_conf = df[df['confidence'] < 0.7]
    low_conf.to_csv('review_queue.csv')  # человек открывает, правит, возвращает
    corrected = pd.read_csv('review_queue_corrected.csv')
    return pd.concat([df[df['confidence'] >= 0.7], corrected])

@task
def select_for_labeling(df): return ActiveLearningAgent().run_cycle(df, n_iterations=5)

@task
def train(df):
    # обучить модель, вернуть метрики
    ...

@flow
def data_pipeline():
    raw = collect()
    clean_df = clean(raw)
    labeled = auto_label(clean_df)
    reviewed = human_review(labeled)    # ← human-in-the-loop
    al_data = select_for_labeling(reviewed)
    train(al_data)

❗ HUMAN-IN-THE-LOOP: ЧТО ИМЕННО ПРОВЕРЯЕТ ЧЕЛОВЕК

- После авторазметки: просматривает примеры с confidence < threshold и исправляет ошибочные метки
- После чистки: проверяет отчёт DataQualityAgent — подтверждает стратегию чистки
- После AL: вручную размечает отобранные агентом примеры
- Перед обучением: просматривает итоговый датасет — подтверждает что данные готовы к обучению

ТРЕБОВАНИЯ К СДАЧЕ

- Единый пайплайн: все 4 агента запускаются одной командой (python run_pipeline.py)
- Human-in-the-loop: минимум 1 явная точка человеческой проверки (не просто запись в лог, а реальная правка)
- Воспроизводимость: README с инструкцией — проверяем на чистом окружении (pip install + python run.py)
- Финальный датасет: data/labeled/ — размечен, чистый, с data card
- Обученная модель + метрики (accuracy/F1 или WER)
- Итоговый отчёт (README или PDF) — 5 разделов

СТРУКТУРА РЕПОЗИТОРИЯ

- agents/ — все 4 агента из заданий 1–4
- run_pipeline.py или pipeline/ — основной файл запуска
- data/raw/ и data/labeled/ — данные
- review_queue.csv — файл для ручной проверки (human-in-the-loop)
- models/ — сохранённая модель
- reports/ — quality_report.md, annotation_report.md, al_report.md
- README.md — инструкция запуска + data card + описание HITL-точки

ФИНАЛЬНЫЙ ОТЧЁТ (README или PDF) — 5 разделов

- 1. Описание задачи и датасета: модальность, объём, классы
- 2. Что делал каждый агент: какие решения приняты и почему
- 3. Описание HITL-точки: сколько примеров проверено, что исправлено
- 4. Метрики качества на каждом этапе + итоговые метрики модели
- 5. Ретроспектива: что сработало, что нет, что бы сделал иначе

КРИТЕРИИ ОЦЕНКИ

- Все 4 агента переиспользованы из заданий 1–4 — 10 баллов
- Пайплайн запускается end-to-end одной командой — 10 баллов
- Есть реальная HITL-точка (не просто лог, а реальная правка данных) — 8 баллов
- Обученная модель + метрики (accuracy/F1/WER) — 7 баллов
- Финальный отчёт: все 5 разделов — 5 баллов

БОНУСЫ

- +3 балла: LLM-агент (Claude API) в пайплайне — например, Claude генерирует спецификацию или объясняет ошибки в данных
- +2 балла: дашборд Streamlit / Gradio — интерфейс для HITL-разметки или визуализации результатов
