from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator

SCRIPTS_DIR = "/opt/airflow/scripts"

default_args = {
    "owner": "airflow",
    "start_date": datetime(2026, 7, 7),
}

with DAG(
    dag_id="nlp_pipeline",
    default_args=default_args,
    schedule="@daily",
    catchup=False,
    description="NLP pipeline: load, clean, sentiment, NER, store",
):

    load_data = BashOperator(
        task_id="load_data",
        bash_command=f"python {SCRIPTS_DIR}/load_data.py",
    )

    clean_data = BashOperator(
        task_id="clean_data",
        bash_command=f"python {SCRIPTS_DIR}/clean_data.py",
    )

    sentiment = BashOperator(
        task_id="sentiment",
        bash_command=f"python {SCRIPTS_DIR}/sentiment.py",
    )

    ner = BashOperator(
        task_id="ner",
        bash_command=f"python {SCRIPTS_DIR}/ner.py",
    )

    store_db = BashOperator(
        task_id="store_db",
        bash_command=f"python {SCRIPTS_DIR}/store_db.py",
    )

    load_data >> clean_data >> sentiment >> ner >> store_db
