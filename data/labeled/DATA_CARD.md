# Data Card — Sentiment Classification Dataset

Generated: 2026-03-26T22:43:41.182942

## Overview

| Field | Value |
|-------|-------|
| Task | Binary sentiment classification |
| Modality | Text |
| Size | 1100 examples |
| Classes | negative, positive |
| Sources | hf_rotten_tomatoes, scrape_https://quotes.toscrape.com |
| Language | English |

## Label Distribution

| Label | Count | Percent |
|-------|-------|---------|
| negative | 607 | 55.2% |
| positive | 493 | 44.8% |

## Columns

- **text**: str
- **label**: str
- **source**: str
- **collected_at**: str
- **predicted_label**: str
- **confidence**: float64
- **needs_review**: bool
- **final_label**: str
- **corrected_label**: str

## Processing Steps

1. Data collected from HuggingFace (rotten_tomatoes) + web scraping (quotes.toscrape.com)
2. Quality check: removed duplicates, fixed missing values, clipped text-length outliers
3. Auto-labeled with confidence scores (zero-shot or TF-IDF fallback)
4. Human-in-the-loop review of low-confidence examples
5. Final labels merged from auto-labels + human corrections

## Known Limitations

- Scraped data (quotes) is not domain-specific for sentiment — adds noise
- Auto-labeling confidence depends on model quality
- HITL coverage is limited to low-confidence subset
