"""
econ_news_agent.py — RSS-based daily economics/finance digest with hybrid topics and GPT summary.
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import feedparser
from dateutil import tz, parser as dateparser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ----------------- HELPERS -----------------

def normalize_datetime(dt_str: str) -> Optional[datetime]:
    """Parse pubDate-ish strings from RSS into timezone-aware UTC datetimes."""
    try:
        dt = dateparser.parse(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz.UTC)
        return dt.astimezone(tz.UTC)
    except Exception:
        return None


def same_calendar_day(dt_utc: Optional[datetime], target_date_local: datetime, local_tz) -> bool:
    """Return True if dt_utc falls on target_date_local when converted to local tz."""
    if dt_utc is None:
        return False
    dt_local = dt_utc.astimezone(local_tz)
    return (dt_local.year, dt_local.month, dt_local.day) == (
        target_date_local.year, target_date_local.month, target_date_local.day
    )


def fetch_feed_entries(feeds) -> list:
    """Fetch & combine RSS entries from the list of feeds."""
    all_entries = []
    for url in feeds:
        d = feedparser.parse(url)
        for e in d.entries:
            title = e.get("title", "").strip()
            link = e.get("link", "").strip()
            published = e.get("published") or e.get("updated") or ""
            published_dt = normalize_datetime(published) if published else None
            summary = e.get("summary", "") or e.get("description", "") or ""
            author = e.get("author", "") or e.get("source", {}).get("title", "")
            tags = []
            if "tags" in e and isinstance(e["tags"], list):
                for t in e["tags"]:
                    term = t.get("term") or t.get("label") or ""
                    if term:
                        tags.append(term.strip())
            all_entries.append({
                "title": title,
                "link": link,
                "published": published,
                "published_dt_utc": published_dt,
                "summary": summary,
                "publisher": author,
                "tags": tags
            })
    return all_entries


def entries_to_dataframe(entries: list, target_date_local: datetime, local_tz) -> pd.DataFrame:
    """Filter entries to the given local date and convert to a DataFrame."""
    filtered = [e for e in entries if same_calendar_day(e["published_dt_utc"], target_date_local, local_tz)]
    df = pd.DataFrame(filtered)
    if df.empty:
        return df
    df["published_local"] = df["published_dt_utc"].apply(
        lambda d: d.astimezone(local_tz) if pd.notnull(d) else None
    )
    df["text_for_cluster"] = (df["title"].fillna("") + " — " + df["summary"].fillna("")).str.strip()
    df["source"] = df["link"].apply(lambda u: u.split("/")[2] if isinstance(u, str) and "://" in u else "")
    df["categories"] = df["tags"].apply(lambda ts: ", ".join(ts) if isinstance(ts, list) and ts else "")
    cols = ["published_local", "title", "link", "publisher", "source",
            "categories", "summary", "tags", "published_dt_utc", "text_for_cluster"]
    df = df[cols]
    df = df.sort_values("published_local")
    return df


# (hybrid_topics, short_bullet_summary, gpt_daily_summary, rss_vs_ml_confusion stay the same)

# ----------------- CLASS API -----------------

@dataclass
class DigestResult:
    date_local: datetime
    df: pd.DataFrame
    total: int
    simple_hybrid: str
    simple_rss: str
    simple_ml: str
    gpt_summary: Optional[str]
    confusion: pd.DataFrame
    topic_maps: Dict[str, Any]


class EconNewsAgent:
    def __init__(self,
                 feeds,
                 local_tz=None,
                 model="gpt-4o-mini",
                 max_k=6,
                 use_rss_threshold=0.4):
        self.feeds = feeds
        self.local_tz = local_tz or tz.UTC
        self.model = model
        self.max_k = max_k
        self.use_rss_threshold = use_rss_threshold

    def build(self, date_str: Optional[str] = None) -> DigestResult:
        if date_str:
            target_date_local = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=self.local_tz)
        else:
            now_local = datetime.now(tz=self.local_tz)
            target_date_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)

        entries = fetch_feed_entries(self.feeds)
        df = entries_to_dataframe(entries, target_date_local=target_date_local, local_tz=self.local_tz)
        total = len(df)

        df, topic_maps = hybrid_topics(df, max_k=self.max_k, use_rss_threshold=self.use_rss_threshold)

        simple_hybrid = short_bullet_summary(df, topic_column="topic")
        simple_rss = short_bullet_summary(df, topic_column="rss_topic")
        simple_ml = short_bullet_summary(df, topic_column="ml_topic")

        gpt = None
        if os.getenv("OPENAI_API_KEY"):
            gpt = gpt_daily_summary(df, date_local=target_date_local)

        cm = rss_vs_ml_confusion(df)

        return DigestResult(
            date_local=target_date_local,
            df=df,
            total=total,
            simple_hybrid=simple_hybrid,
            simple_rss=simple_rss,
            simple_ml=simple_ml,
            gpt_summary=gpt,
            confusion=cm,
            topic_maps=topic_maps
        )


    # ------- pretty printers / utilities you can call separately -------

    def print_overviews(self, res: DigestResult):
        print(f"\nDate (local): {res.date_local.date()} | Items: {res.total}")
        print("\n== Hybrid overview (topic) ==\n" + res.simple_hybrid)
        print("\n== RSS overview (rss_topic) ==\n" + res.simple_rss)
        print("\n== ML overview (ml_topic) ==\n" + res.simple_ml)
        if res.gpt_summary:
            print("\n== GPT daily wrap ==\n" + (res.gpt_summary or ""))

    def print_rows(self, res: DigestResult, limit: int = 20):
        if res.total == 0:
            print("(No rows)")
            return
        view_cols = [
            "published_local", "title", "source",
            "rss_topic", "rss_topic_count",
            "ml_topic", "ml_topic_count", "ml_strength",
            "topic", "link"
        ]
        print(f"\n== Data (first {min(limit, res.total)} rows) ==\n")
        print(res.df[view_cols].head(limit).to_string(index=False))

    def print_confusion(self, res: DigestResult):
        if res.confusion.empty:
            print("(No confusion table — not enough data)")
        else:
            print("\n== RSS vs ML confusion table ==\n")
            print(res.confusion.to_string())

    def to_csv(self, res: DigestResult, path: str):
        res.df.to_csv(path, index=False)


# ----------------- CLI (optional) -----------------

def _cli():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="Target date in YYYY-MM-DD (local tz). Default: today.")
    args = ap.parse_args()

    agent = EconNewsAgent()
    res = agent.build(args.date)
    agent.print_overviews(res)
    agent.print_rows(res, limit=20)
    agent.print_confusion(res)


if __name__ == "__main__":
    _cli()
