import re
import sys
import pandas as pd
from pathlib import Path
from camel_tools.utils.normalize import normalize_alef_ar, normalize_alef_maksura_ar, normalize_teh_marbuta_ar
from camel_tools.utils.dediac import dediac_ar

sys.stdout.reconfigure(encoding="utf-8")

project_root = Path(__file__).resolve().parent.parent
input_file = project_root / "data" / "scraped_posts.csv"
output_file = project_root / "data" / "cleaned_posts.csv"

df = pd.read_csv(input_file)

emoji_pattern = re.compile(
    "[\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U0000FE00-\U0000FE0F"
    "\U0000200D"
    "\U00002B50"
    "\U00002300-\U000023FF"
    "]+", flags=re.UNICODE
)

url_pattern = re.compile(r"https?://\S+|www\.\S+")
mention_pattern = re.compile(r"@\w+")
hashtag_symbol_pattern = re.compile(r"#(\w+)")

def clean_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = url_pattern.sub("", text)
    text = mention_pattern.sub("", text)
    text = hashtag_symbol_pattern.sub(r"\1", text)
    text = emoji_pattern.sub("", text)
    text = normalize_alef_ar(text)
    text = normalize_alef_maksura_ar(text)
    text = normalize_teh_marbuta_ar(text)
    text = dediac_ar(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["text"].apply(clean_text)

output_file.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_file, index=False, encoding="utf-8-sig")

for i in range(min(5, len(df))):
    orig = df.loc[i, "text"]
    clean = df.loc[i, "clean_text"]
    print(f"--- Row {i} (orig={len(orig)} chars, clean={len(clean)} chars) ---")
    print(f"Original: {orig}")
    print(f"Cleaned:  {clean}")
    print()
