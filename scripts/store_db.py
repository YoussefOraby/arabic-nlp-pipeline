import ast
import json
import sqlite3
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
input_file = project_root / "data" / "final_posts.csv"
db_file = project_root / "data" / "pipeline.db"

df = pd.read_csv(input_file)

db_file.parent.mkdir(parents=True, exist_ok=True)
conn = sqlite3.connect(str(db_file))
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS posts")
cursor.execute("""
    CREATE TABLE posts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT,
        clean_text TEXT,
        sentiment TEXT,
        confidence REAL,
        entities TEXT
    )
""")
conn.commit()

data = []
for _, row in df.iterrows():
    raw = row["entities"]
    if isinstance(raw, str):
        parsed = ast.literal_eval(raw)
    else:
        parsed = raw
    data.append((
        row["text"],
        row["clean_text"],
        row["sentiment"],
        row["confidence"],
        json.dumps(parsed, ensure_ascii=False),
    ))

cursor.executemany(
    "INSERT INTO posts (text, clean_text, sentiment, confidence, entities) VALUES (?, ?, ?, ?, ?)",
    data,
)
conn.commit()

total = cursor.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
print(f"Inserted {total} rows")

result = cursor.execute(
    "SELECT sentiment, COUNT(*) FROM posts GROUP BY sentiment"
).fetchall()

print()
for sentiment, count in result:
    print(f"{sentiment}: {count}")

conn.close()
