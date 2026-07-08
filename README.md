# Arabic NLP Pipeline

An end-to-end NLP pipeline for processing Arabic text, spanning data collection, cleaning, sentiment analysis, named entity recognition, and visualization.

## Project Structure

```
├── dags/                  # Airflow DAG files
├── data/                  # CSV, SQLite outputs
├── notebooks/             # Jupyter experiments
├── scripts/               # Python pipeline scripts
├── config/                # Airflow config
├── logs/                  # Airflow logs
├── plugins/               # Airflow plugins
├── .env                   # Airflow environment variables
├── .gitignore
├── Dockerfile             # Airflow custom image
├── docker-compose.yaml    # Airflow services
└── requirements.txt
```

## Pipeline Stages

Each stage is a standalone script in `scripts/`.

### Data Collection (choose one)

- **`load_data.py`** — Loads the static `komari6/ajgt_twitter_ar` dataset from Hugging Face (1800 Arabic tweets with sentiment labels). Output: `data/scraped_posts.csv`.

- **`load_live_data.py`** — Pulls live comments from YouTube via the YouTube Data API v3. Searches for videos matching a topic keyword and collects up to 50 comments per video (up to 10 videos). Output: `data/scraped_posts.csv`.

### Cleaning

- **`clean_data.py`** — Reads `scraped_posts.csv`, normalizes Arabic characters (camel-tools), removes URLs/mentions/hashtag symbols/emojis/extra whitespace. Output: `data/cleaned_posts.csv`.

### Sentiment Analysis

- **`sentiment.py`** — Runs `CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment` via Hugging Face transformers pipeline on the cleaned text. Output: `data/sentiment_posts.csv` with `sentiment` and `confidence` columns.

### Named Entity Recognition

- **`ner.py`** — Runs `CAMeL-Lab/bert-base-arabic-camelbert-mix-ner` to extract entities (LOC, ORG, PERS, MISC) from the cleaned text. Output: `data/final_posts.csv` with an `entities` column.

### Storage

- **`store_db.py`** — Writes the final data into a SQLite database at `data/pipeline.db`.

### Dashboard

- **`dashboard.py`** — A Plotly Dash web app (port 8050) showing sentiment distribution, top entities, and recent posts.

## Data Sources

### Static Dataset (default)

The `komari6/ajgt_twitter_ar` dataset contains 1,800 Arabic Jordanian tweets with positive/negative/neutral sentiment labels.

### Live Data Source (YouTube)

`load_live_data.py` fetches real-time comments from YouTube using the **YouTube Data API v3**. It requires a free API key:

1. Get a key from the [Google Cloud Console](https://console.cloud.google.com/).
2. Enable the **YouTube Data API v3**.
3. Set the key as an environment variable before running:

```powershell
$env:YOUTUBE_API_KEY = "your_api_key_here"
python scripts/load_live_data.py
```

The topic keyword is configurable at the top of the script (`TOPIC = "فودافون مصر"`). Comments from videos with comments disabled are skipped automatically.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running Scripts

Activate the virtual environment first, then run any script:

```powershell
.\venv\Scripts\Activate.ps1
python scripts/load_data.py      # or load_live_data.py
python scripts/clean_data.py
python scripts/sentiment.py
python scripts/ner.py
python scripts/store_db.py
python scripts/dashboard.py      # opens at http://localhost:8050
```

## Airflow Orchestration

The project includes a Docker-based Airflow setup for scheduled runs.

```powershell
docker compose build --no-cache
docker compose up airflow-init
docker compose up -d
```

The DAG `nlp_pipeline` runs daily and executes all pipeline stages in order. Access the UI at `http://localhost:8081` (user: `airflow`, password: `airflow`).
