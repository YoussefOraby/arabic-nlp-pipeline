import json
import sqlite3
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html
from dash.dash_table import DataTable

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "pipeline.db"

conn = sqlite3.connect(str(DB_PATH))
df = pd.read_sql("SELECT * FROM posts", conn)
conn.close()

sentiment_counts = df["sentiment"].value_counts().reset_index()
sentiment_counts.columns = ["sentiment", "count"]
pie = px.pie(sentiment_counts, names="sentiment", values="count", title="Sentiment Distribution")

entity_counter = Counter()
for row in df["entities"]:
    for entity_text, _ in json.loads(row):
        entity_counter[entity_text] += 1
top_entities = entity_counter.most_common(10)
entities_df = pd.DataFrame(top_entities, columns=["entity", "count"])
bar = px.bar(entities_df, x="count", y="entity", orientation="h", title="Top 10 Entities")
bar.update_layout(yaxis={"categoryorder": "total ascending"})

recent = df.tail(20)[["clean_text", "sentiment", "confidence"]].to_dict("records")

app = Dash(__name__)
app.layout = html.Div(
    [
        html.H1("Arabic NLP Sentiment Dashboard"),
        dcc.Graph(figure=pie),
        dcc.Graph(figure=bar),
        html.H2("Recent Posts"),
        DataTable(
            columns=[
                {"name": "Text", "id": "clean_text"},
                {"name": "Sentiment", "id": "sentiment"},
                {"name": "Confidence", "id": "confidence", "type": "numeric", "format": {"specifier": ".4f"}},
            ],
            data=recent,
            page_size=20,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "maxWidth": "400px", "overflow": "hidden", "textOverflow": "ellipsis"},
        ),
    ],
    style={"padding": "20px", "fontFamily": "sans-serif"},
)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
