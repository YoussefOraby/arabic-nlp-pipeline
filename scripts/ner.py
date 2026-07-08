import sys
import pandas as pd
from pathlib import Path
from transformers import pipeline, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")

project_root = Path(__file__).resolve().parent.parent
input_file = project_root / "data" / "sentiment_posts.csv"
output_file = project_root / "data" / "final_posts.csv"

df = pd.read_csv(input_file)
df = df[df["clean_text"].notna() & (df["clean_text"].str.strip() != "")].reset_index(drop=True)

ner_tokenizer = AutoTokenizer.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-mix-ner")
ner_pipe = pipeline(
    "token-classification",
    model="CAMeL-Lab/bert-base-arabic-camelbert-mix-ner",
    tokenizer=ner_tokenizer,
    aggregation_strategy="none",
)

def group_entities(tokens):
    grouped = []
    cur_wid = None
    cur_type = None
    cur_parts = []
    for t in tokens:
        label = t["entity"]
        if label == "O":
            if cur_type:
                grouped.append(("".join(cur_parts).replace("##", ""), cur_type))
                cur_type = None
                cur_parts = []
                cur_wid = None
            continue
        etype = label.split("-", 1)[1] if "-" in label else label
        wid = t.get("word_id")
        if wid is not None and wid == cur_wid and etype == cur_type:
            cur_parts.append(t["word"])
        else:
            if cur_type:
                grouped.append(("".join(cur_parts).replace("##", ""), cur_type))
            cur_wid = wid
            cur_type = etype
            cur_parts = [t["word"]]
    if cur_type:
        grouped.append(("".join(cur_parts).replace("##", ""), cur_type))
    return grouped

all_entities = []
for text in df["clean_text"]:
    enc = ner_tokenizer(text, truncation=True, max_length=512)
    word_ids = enc.word_ids()
    truncated = ner_tokenizer.decode(enc["input_ids"], skip_special_tokens=True)
    results = ner_pipe(truncated)
    for r in results:
        tid = r.get("index")
        r["word_id"] = word_ids[tid] if tid is not None and tid < len(word_ids) else None
    all_entities.append(group_entities(results))

df["entities"] = all_entities

output_file.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_file, index=False, encoding="utf-8-sig")

for i in range(min(5, len(df))):
    print(f"--- Row {i} ---")
    print(f"Text:     {df.loc[i, 'clean_text'][:80]}")
    print(f"Entities: {df.loc[i, 'entities']}")
    print()
