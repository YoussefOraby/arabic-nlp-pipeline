# Arabic NLP Pipeline: Sentiment + NER on Social Media

An end-to-end, fully automated NLP pipeline that collects Arabic/English social media text, classifies sentiment, extracts named entities, and visualizes the results on a live dashboard — orchestrated with Apache Airflow and containerized with Docker.

## Overview

This project processes Arabic social media posts through a complete data pipeline:

```
Load/Scrape Data → Clean & Normalize Text → Sentiment Classification → Entity Extraction (NER) → Store in Database → Live Dashboard
```

The entire pipeline runs automatically on a daily schedule, with zero manual intervention required after setup.

## Architecture

| Stage | Tool | Purpose |
|---|---|---|
| Data Collection | Hugging Face Datasets | Loads real-world Arabic Twitter data |
| Text Cleaning | CAMeL Tools | Normalizes Arabic script, removes noise |
| Sentiment Analysis | CAMeL-Lab AraBERT (via Hugging Face Transformers) | Classifies posts as positive/negative/neutral |
| Named Entity Recognition | CAMeL-Lab AraBERT NER (via Hugging Face Transformers) | Tags people, organizations, and locations |
| Storage | SQLite | Persists processed results |
| Orchestration | Apache Airflow (Docker) | Automates and schedules the full pipeline daily |
| Visualization | Plotly Dash | Live dashboard showing sentiment trends and top entities |

## Tech Stack

- **Language:** Python 3.13
- **ML/NLP:** Hugging Face Transformers, PyTorch (CPU), CAMeL Tools
- **Data:** Pandas, SQLite
- **Orchestration:** Apache Airflow 3.3.0, Docker Compose
- **Dashboard:** Plotly Dash

## Pipeline Stages

1. **`load_data.py`** — Loads a labeled Arabic Twitter sentiment dataset (~1,800 posts) as the data source.
2. **`clean_data.py`** — Normalizes Arabic characters, strips URLs/mentions/emojis, and prepares text for modeling.
3. **`sentiment.py`** — Runs each post through a pretrained AraBERT sentiment model, producing a label and confidence score.
4. **`ner.py`** — Extracts named entities (locations, organizations, etc.) from each post using a pretrained AraBERT NER model.
5. **`store_db.py`** — Writes the final, enriched dataset into a SQLite database.
6. **`dashboard.py`** — Serves a live web dashboard visualizing sentiment distribution, top mentioned entities, and recent posts.

All five processing stages are chained together and automated via an **Airflow DAG** (`nlp_pipeline`), running daily via Docker Compose.

## Running the Project

### Prerequisites
- Docker Desktop
- Python 3.13 (for local dashboard/dev use)

### 1. Set up the pipeline (Airflow + Docker)
```bash
docker compose build --no-cache
docker compose up airflow-init
docker compose up -d
```

Airflow UI: [http://localhost:8081](http://localhost:8081) (default login: `airflow` / `airflow`)

### 2. Trigger the pipeline
From the Airflow UI, unpause and trigger the `nlp_pipeline` DAG — or wait for its daily scheduled run.

### 3. View the dashboard
```bash
python scripts/dashboard.py
```
Then open [http://localhost:8050](http://localhost:8050)

## Results

- **Dataset size:** ~1,800 Arabic social media posts
- **Sentiment distribution:** 837 positive / 836 negative / 127 neutral
- **Pipeline runtime:** ~24 minutes end-to-end (CPU-only inference)

## Notes

- All models used are pretrained (no fine-tuning required for this version) — see "Future Work" below for planned improvements.
- Torch is installed as a CPU-only build to keep the Docker image lightweight and avoid unnecessary GPU/CUDA dependencies.

## Future Work

- Fine-tune AraBERT on domain-specific labeled data for improved accuracy
- Add live social media scraping as a data source
- Expand the dashboard with time-series sentiment trends
- Add authentication and deploy the dashboard publicly

## Author

Youssef Oraby
