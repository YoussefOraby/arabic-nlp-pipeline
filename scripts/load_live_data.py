import os
import sys
import csv
from pathlib import Path

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv()

if len(sys.argv) < 2:
    print("Usage: python scripts/load_live_data.py \"<search topic>\"")
    sys.exit(1)

TOPIC = sys.argv[1]
MAX_VIDEOS = 10
MAX_COMMENTS_PER_VIDEO = 50

api_key = os.environ["YOUTUBE_API_KEY"]
youtube = build("youtube", "v3", developerKey=api_key)

output_dir = Path(__file__).resolve().parent.parent / "data"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "scraped_posts.csv"

search = youtube.search().list(
    q=TOPIC,
    part="snippet",
    type="video",
    maxResults=MAX_VIDEOS,
    relevanceLanguage="ar",
).execute()

video_titles = {}
for item in search.get("items", []):
    video_titles[item["id"]["videoId"]] = item["snippet"]["title"]

comments = []
for vid in video_titles:
    try:
        req = youtube.commentThreads().list(
            part="snippet",
            videoId=vid,
            maxResults=MAX_COMMENTS_PER_VIDEO,
        )
        while req and len(comments) < len(video_titles) * MAX_COMMENTS_PER_VIDEO:
            res = req.execute()
            for item in res.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "text": snippet["textDisplay"],
                    "video_title": video_titles[vid],
                    "author": snippet["authorDisplayName"],
                    "date": snippet["publishedAt"],
                })
            req = youtube.commentThreads().list_next(req, res)
    except HttpError as e:
        if e.resp.status == 403:
            pass
        else:
            raise

with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=["text", "video_title", "author", "date"])
    writer.writeheader()
    writer.writerows(comments)

print(f"Collected {len(comments)} comments")
