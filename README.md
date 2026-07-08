# Arabic NLP Pipeline: Sentiment + NER on Social Media

An end-to-end, fully automated NLP pipeline that collects Arabic/English social media text, classifies sentiment, extracts named entities, and visualizes the results on a live dashboard — orchestrated with Apache Airflow and containerized with Docker.

## Overview

This project processes Arabic social media posts through a complete data pipeline:

```
Load/Scrape Data → Clean & Normalize Text → Sentiment Classification → Entity Extraction (NER) → Store in Database → Live Dashboard
```

The entire pipeline runs automatically on a daily schedule, with zero manual intervention required after setup.

## Project Structure

```
├── dags/                  # Airflow DAG files
├── data/                  # CSV, SQLite outputs
├── notebooks/             # Jupyter experiments
├── scripts/               # Python pipeline scripts
├── config/                # Airflow config
├── logs/                  # Airflow logs
├── plugins/               # Airflow plugins
├── .env                   # Environment variables (API keys, Airflow)
├── .env.example           # Example env file (safe to commit)
├── .gitignore
├── Dockerfile             # Airflow custom image
├── docker-compose.yaml    # Airflow services
└── requirements.txt
```

## Architecture

| Stage | Tool | Purpose |
|---|---|---|
| Data Collection | Hugging Face Datasets / YouTube Data API | Loads real-world Arabic Twitter data or live YouTube comments |
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

Each stage is a standalone script in `scripts/`.

### Data Collection (choose one)

- **`load_data.py`** — Loads a labeled Arabic Twitter sentiment dataset from Hugging Face (`komari6/ajgt_twitter_ar`, ~1,800 posts). Output: `data/scraped_posts.csv`.
- **`load_live_data.py`** — Pulls live comments from YouTube via the YouTube Data API v3. Searches for videos matching a topic keyword and collects up to 50 comments per video (up to 10 videos). Output: `data/scraped_posts.csv`.

### Cleaning

- **`clean_data.py`** — Decodes HTML entities, strips HTML tags, normalizes Arabic characters (camel-tools), removes URLs/mentions/hashtag symbols/emojis, and trims whitespace. Output: `data/cleaned_posts.csv`.

### Sentiment Analysis

- **`sentiment.py`** — Runs `CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment` via Hugging Face transformers pipeline on each post, producing a label and confidence score. Output: `data/sentiment_posts.csv`.

### Named Entity Recognition

- **`ner.py`** — Runs `CAMeL-Lab/bert-base-arabic-camelbert-mix-ner` to extract entities (locations, organizations, persons, miscellaneous). Output: `data/final_posts.csv` with an `entities` column.

### Storage

- **`store_db.py`** — Writes the final, enriched dataset into a SQLite database at `data/pipeline.db`.

### Dashboard

- **`dashboard.py`** — Serves a live Plotly Dash web dashboard (port 8050) visualizing sentiment distribution, top mentioned entities, and recent posts.

All processing stages are chained together and automated via an **Airflow DAG** (`nlp_pipeline`), running daily via Docker Compose.

## Data Sources

### Static Dataset (default)

The `komari6/ajgt_twitter_ar` dataset contains 1,800 Arabic Jordanian tweets with positive/negative/neutral sentiment labels.

### Live Data Source (YouTube)

`load_live_data.py` fetches real-time comments from YouTube using the **YouTube Data API v3**. It requires a free API key:

1. Get a key from the [Google Cloud Console](https://console.cloud.google.com/).
2. Enable the **YouTube Data API v3**.
3. Create a `.env` file in the project root with your key (use `.env.example` as a template):

```
YOUTUBE_API_KEY=AIzaSy...
```

The key is loaded automatically via `python-dotenv` — no need to set it manually each session. The `.env` file is listed in `.gitignore` and will never be committed.

The topic keyword is passed as a command-line argument. Comments from videos with comments disabled are skipped automatically.

```powershell
python scripts/load_live_data.py "فودافون مصر"
```

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

### Local Development (without Docker)

Activate the virtual environment and run scripts individually:

```powershell
.\venv\Scripts\Activate.ps1
python scripts/load_data.py      # or load_live_data.py
python scripts/clean_data.py
python scripts/sentiment.py
python scripts/ner.py
python scripts/store_db.py
python scripts/dashboard.py
```

## Results

- **Dataset size:** ~1,800 Arabic social media posts
- **Sentiment distribution:** 837 positive / 836 negative / 127 neutral
- **Pipeline runtime:** ~24 minutes end-to-end (CPU-only inference)

## Notes

- All models used are pretrained (no fine-tuning required for this version) — see "Future Work" below for planned improvements.
- Torch is installed as a CPU-only build to keep the Docker image lightweight and avoid unnecessary GPU/CUDA dependencies.

## Future Work

- Fine-tune AraBERT on domain-specific labeled data for improved accuracy
- Expand the dashboard with time-series sentiment trends
- Add authentication and deploy the dashboard publicly

## Author

Youssef Oraby
