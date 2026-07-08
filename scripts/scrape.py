import snscrape.modules.twitter as sntwitter
import csv
from pathlib import Path

query = "فودافون مصر"
limit = 100
data_dir = Path(__file__).resolve().parent.parent / "data"
data_dir.mkdir(parents=True, exist_ok=True)
output_file = data_dir / "scraped_posts.csv"

posts = []

try:
    scraper = sntwitter.TwitterSearchScraper(query)
    for i, tweet in enumerate(scraper.get_items()):
        if i >= limit:
            break
        posts.append({
            "text": tweet.content,
            "date": tweet.date.isoformat(),
            "username": tweet.username,
            "like_count": tweet.likeCount,
        })
except Exception as e:
    print(f"Error during scraping: {e}")

if not posts:
    print("No posts found for query: فودافون مصر")
else:
    with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "date", "username", "like_count"])
        writer.writeheader()
        writer.writerows(posts)
    print(f"Saved {len(posts)} posts to {output_file}")
