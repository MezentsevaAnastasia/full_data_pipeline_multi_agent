Задание 4 — ALAgent / MultimodalAgent | 15 баллов | Дедлайн: занятие 9 (27.03.2026)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ЦЕЛЬ

Построить агента для умного отбора данных (Active Learning) или сборки мультимодального датасета. Студент выбирает один трек. Агент войдёт в финальный пайплайн как active_learning_op.

━━━ ТРЕК A: ActiveLearningAgent ━━━

АРХИТЕКТУРА

- ActiveLearningAgent
- skill: fit(labeled_df) → model (обучить базовую модель)
- skill: query(pool, strategy) → indices (стратегии: 'entropy', 'margin', 'random')
- skill: evaluate(labeled_df, test_df) → Metrics (accuracy, F1)
- skill: report(history) → LearningCurve (график quality vs. n_labeled)

ТЕХНИЧЕСКИЙ КОНТРАКТ

from al_agent import ActiveLearningAgent

agent = ActiveLearningAgent(model='logreg')

# Цикл: старт с N=50, 5 итераций по 20 примеров
history = agent.run_cycle(
    labeled_df=df_labeled_50,
    pool_df=df_unlabeled,
    strategy='entropy',
    n_iterations=5,
    batch_size=20
)

# → history: список {iteration, n_labeled, accuracy, f1}
agent.report(history)  # → learning_curve.png

ЧТО СДАТЬ (ТРЕК A)

- AL-цикл: старт с N=50 → 5 итераций → финальная модель
- Сравнение стратегий: entropy vs random — кривые обучения на одном графике
- Вывод: сколько примеров сэкономлено при том же качестве (accuracy/F1) vs random baseline
- Файл: agents/al_agent.py + notebooks/al_experiment.ipynb

━━━ ТРЕК B: MultimodalAgent ━━━

АРХИТЕКТУРА

- MultimodalAgent
- skill: load_modality(path, type) → ModalityData
- skill: align(modalities: dict, key) → AlignedDataset (выравнивание по общему ID или timestamp)
- skill: describe(aligned_df) → EDAReport (распределения, примеры, покрытие)
- skill: export(aligned_df, format) → file (csv / parquet / hf dataset)

ЧТО СДАТЬ (ТРЕК B)

- Датасет с 2+ модальностями (text+audio, text+image, audio+image)
- Выравнивание по общему ключу (ID / timestamp) — без потери данных
- EDA: совместное распределение меток по модальностям, пример на каждую пару
- Описание ML-задачи: как этот датасет будет использован?

КРИТЕРИИ ОЦЕНКИ (оба трека)

- Агент работает, скиллы реализованы — 8 баллов
- Анализ / сравнение / визуализации — 4 балла
- README + воспроизводимость (pip install -r requirements.txt && python run.py) — 2 балла
- Бонус: LLM-скилл (Claude API) в пайплайне — +1 балл