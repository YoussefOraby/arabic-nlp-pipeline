FROM apache/airflow:3.3.0

USER root
RUN apt-get update && apt-get install -y --no-install-recommends build-essential g++ && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install snscrape pandas transformers camel-tools datasets
