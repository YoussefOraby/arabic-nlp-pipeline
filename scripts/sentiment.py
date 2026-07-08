import sys
import pandas as pd
from pathlib import Path
from transformers import pipeline

sys.stdout.reconfigure(encoding="utf-8")

project_root = Path(__file__).resolve().parent.parent
input_file = project_root / "data" / "cleaned_posts.csv"
output_file = project_root / "data" / "sentiment_posts.csv"

df = pd.read_csv(input_file)
df = df[df["clean_text"].notna() & (df["clean_text"].str.strip() != "")].reset_index(drop=True)

pipe = pipeline(
    "text-classification",
    model="CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment",
    truncation=True,
)

results = pipe(df["clean_text"].tolist(), batch_size=32)
df["sentiment"] = [r["label"] for r in results]
df["confidence"] = [round(r["score"], 4) for r in results]

output_file.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(df["sentiment"].value_counts().to_string())
print()
for i in range(min(5, len(df))):
    print(f"--- Row {i} ---")
    print(f"Text:       {df.loc[i, 'clean_text'][:80]}")
    print(f"Sentiment:  {df.loc[i, 'sentiment']}")
    print(f"Confidence: {df.loc[i, 'confidence']}")
    print()
