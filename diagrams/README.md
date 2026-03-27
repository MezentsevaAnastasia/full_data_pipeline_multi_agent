# Pipeline Diagrams

Все схемы оформлены в формате Mermaid — рендерятся в GitHub, VS Code (расширение Mermaid Preview) и Obsidian.

## Содержание

| Файл | Описание |
|------|----------|
| [00_pipeline_overview.md](00_pipeline_overview.md) | **Общая схема** — 7 шагов пайплайна end-to-end |
| [01_data_collection_agent.md](01_data_collection_agent.md) | **DataCollectionAgent** — источники данных, merge, дедупликация |
| [02_data_quality_agent.md](02_data_quality_agent.md) | **DataQualityAgent** — обнаружение и исправление проблем |
| [03_annotation_agent.md](03_annotation_agent.md) | **AnnotationAgent** — авто-разметка, BART, fallback, экспорт |
| [04_hitl_workflow.md](04_hitl_workflow.md) | **HITL Workflow** — Human-in-the-Loop процесс ревью |
| [05_active_learning_agent.md](05_active_learning_agent.md) | **ActiveLearningAgent** — стратегии выборки, итерации, сравнение |
| [06_data_flow.md](06_data_flow.md) | **Data Flow** — файлы и артефакты на каждом шаге |
| [07_model_training.md](07_model_training.md) | **Model Training** — финальное обучение и метрики |

## Общая схема пайплайна

```
config.yaml
    │
    ▼
[STEP 1] DataCollectionAgent
    │  HuggingFace + Web Scraping → Merge
    │  → data/raw/pipeline_raw.csv
    ▼
[STEP 2] DataQualityAgent
    │  detect_issues → fix → compare
    │  → data/raw/pipeline_clean.csv
    ▼
[STEP 3] AnnotationAgent
    │  BART zero-shot (или TF-IDF fallback)
    │  predicted_label + confidence + needs_review
    │  → reports/annotation_spec.md
    │  → reports/labelstudio_import.json
    ▼
[STEP 4] Human-in-the-Loop
    │  Низкая уверенность → review_queue.csv
    │  Human (или авто-симуляция) → corrected_label
    │  Merge → final_label
    │  → data/labeled/pipeline_labeled.csv
    ▼
[STEP 5] ActiveLearningAgent
    │  Стратегии: entropy vs random
    │  5 итераций × 20 примеров
    │  → reports/learning_curve.png
    │  → reports/al_report.md
    ▼
[STEP 6] Model Training
    │  TF-IDF → SVD → Normalizer → LogReg
    │  → models/final_model.pkl
    ▼
[STEP 7] Report Generation
    │  → reports/final_report.md
    │  → data/labeled/DATA_CARD.md
    ▼
   ✅ Done
```

## Как смотреть

- **GitHub** — откройте любой `.md` файл, Mermaid рендерится автоматически
- **VS Code** — установите расширение `Mermaid Preview` или `Markdown Preview Mermaid Support`
- **Obsidian** — поддерживается из коробки
