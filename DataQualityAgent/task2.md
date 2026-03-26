Задание 2 — DataQualityAgent «Детектив данных» | 15 баллов | Дедлайн: занятие 7 (20.03.2026)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ЦЕЛЬ

Написать агента-детектива, который автоматически выявляет и устраняет проблемы качества данных. 

АРХИТЕКТУРА АГЕНТА

- DataQualityAgent
- skill: detect_issues(df) → QualityReport (missing values, duplicates, outliers, class imbalance)
- skill: fix(df, strategy: dict) → DataFrame
- skill: compare(df_before, df_after) → ComparisonReport

ТЕХНИЧЕСКИЙ КОНТРАКТ

from data_quality_agent import DataQualityAgent

agent = DataQualityAgent()
report = agent.detect_issues(df)
# → {'missing': {...}, 'duplicates': N, 'outliers': [...], 'imbalance': {...}}

df_clean = agent.fix(df, strategy={
    'missing': 'median',
    'duplicates': 'drop',
    'outliers': 'clip_iqr'
})

comparison = agent.compare(df, df_clean)
# → таблица: было / стало по каждой метрике

ТРЕБОВАНИЯ

- Минимум 3 типа проблем: пропущенные значения, дубликаты, выбросы (IQR или z-score)
- Минимум 2 стратегии чистки на выбор — параметр strategy в методе fix()
- Сравнительный отчёт до/после по каждой метрике качества
- Обоснование выбранной стратегии — Markdown-ячейка в ноутбуке

ТРИ ЧАСТИ ЗАДАНИЯ

- Часть 1: Детектив — обнаружить пропуски, выбросы, дубли, дисбаланс классов. Визуализировать каждую проблему.
- Часть 2: Хирург — применить минимум 2 стратегии чистки, сравнить результаты в таблице.
- Часть 3: Аргумент — обосновать выбор лучшего подхода: почему эта стратегия лучше для вашей ML-задачи?

БОНУС (+2 балла): Реализовать LLM-скилл — Claude API объясняет найденные проблемы и рекомендует стратегию чистки на основе описания задачи.