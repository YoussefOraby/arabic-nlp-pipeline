from datasets import load_dataset
import pandas as pd
from pathlib import Path

output_dir = Path(__file__).resolve().parent.parent / "data"
output_dir.mkdir(parents=True, exist_ok=True)

ds = load_dataset("komari6/ajgt_twitter_ar", split="train")
df = pd.DataFrame(ds)
output_file = output_dir / "scraped_posts.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"Loaded {len(df)} rows")
