"""
econ_news_agent.py — RSS-based daily economics/finance digest with hybrid topics and GPT summary.

This module deliberately avoids hard-coding FEEDS / LOCAL_TZ / OPENAI_MODEL.
Pass them in when constructing EconNewsAgent.

- feeds: list[str] of RSS/Atom feed URLs (required)
- local_tz: dateutil.tz timezone object (defaults to UTC if not provided)
- model: OpenAI model name for summaries (defaults to "gpt-4o-mini" if not provided)
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime

import numpy as np
import pandas as pd
import feedparser
from dateutil import tz, parser as dateparser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

__all__ = [
    "EconNewsAgent", "DigestResult",
    "normalize_datetime", "same_calendar_day",
    "fetch_feed_entries", "entries_to_dataframe",
    "hybrid_topics", "short_bullet_summary",
    "gpt_daily_summary", "rss_vs_ml_confusion",
]


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


def fetch_feed_entries(feeds: List[str]) -> list:
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
            # RSS categories/tags (list of dicts with 'term' in feedparser)
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
    # For convenience columns
    df["published_local"] = df["published_dt_utc"].apply(lambda d: d.astimezone(local_tz) if pd.notnull(d) else None)
    # Consolidate preview text for clustering
    df["text_for_cluster"] = (df["title"].fillna("") + " — " + df["summary"].fillna("")).str.strip()
    # Add a simple “source” column from the link’s netloc
    df["source"] = df["link"].apply(lambda u: u.split("/")[2] if isinstance(u, str) and "://" in u else "")
    # Make a printable categories string (if present)
    df["categories"] = df["tags"].apply(lambda ts: ", ".join(ts) if isinstance(ts, list) and ts else "")
    # Order columns
    cols = ["published_local", "title", "link", "publisher", "source", "categories", "summary",
            "tags", "published_dt_utc", "text_for_cluster"]
    df = df[cols]
    df = df.sort_values("published_local")
    return df


def hybrid_topics(df: pd.DataFrame, max_k: int = 6, use_rss_threshold: float = 0.4) -> Tuple[pd.DataFrame, Dict[str, Dict[int, str]]]:
    """
    Keep RSS categories and also compute KMeans clusters on TF-IDF.

    Adds columns:
      - rss_topic           (str): first RSS category (or "Uncategorized")
      - rss_topic_count     (int): frequency of that rss_topic in the day
      - ml_cluster_id       (int): KMeans cluster id
      - ml_topic            (str): short label from top TF-IDF terms for that cluster
      - ml_topic_count      (int): frequency of that ml_topic in the day
      - ml_strength         (float): closeness to centroid (higher ~ more central)
      - topic               (str): “hybrid” display label (prefers RSS if prevalent)

    Returns: df, {"rss_labels": {...}, "ml_labels": {...}}
    """
    if df.empty:
        return df, {"rss_labels": {}, "ml_labels": {}}

    df = df.copy()

    # --- 1) RSS categories -> rss_topic
    def first_cat(s: str) -> str:
        if isinstance(s, str) and s.strip():
            return s.split(",")[0].strip()
        return "Uncategorized"

    df["rss_topic"] = df["categories"].apply(first_cat).str.strip()
    rss_norm = df["rss_topic"].str.lower().str.strip()
    rss_counts = rss_norm.value_counts().to_dict()
    rss_labels = {t: t.title() for t in rss_counts.keys()}
    df["rss_topic_count"] = rss_norm.map(rss_counts)
    df["rss_topic"] = rss_norm.map(rss_labels)

    # --- 2) TF-IDF + KMeans -> ml_cluster_id / ml_topic (+ strength)
    texts = df["text_for_cluster"].fillna("").tolist()

    vectorizer = TfidfVectorizer(max_features=4000, ngram_range=(1, 2), stop_words="english")
    X = vectorizer.fit_transform(texts)

    n = len(texts)
    if n <= 2:
        df["ml_cluster_id"] = 0
        ml_labels = {0: "Misc"}
        df["ml_topic"] = "Misc"
        df["ml_strength"] = 1.0
    else:
        if n <= 5:
            k = 1
        elif n <= 12:
            k = 2
        elif n <= 25:
            k = 3
        else:
            k = min(max_k, max(3, n // 12))

        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        clusters = km.fit_predict(X)
        df["ml_cluster_id"] = clusters

        # label clusters by top TF-IDF terms
        terms = vectorizer.get_feature_names_out()
        centers = km.cluster_centers_
        ml_labels = {}
        for i in range(k):
            top_idx = centers[i].argsort()[::-1][:4]
            label = ", ".join([terms[j] for j in top_idx])
            ml_labels[i] = label.title()
        df["ml_topic"] = df["ml_cluster_id"].map(ml_labels)

        # strength = 1/(1+distance to assigned centroid)
        dists = km.transform(X)  # (n_docs, k)
        assigned = df["ml_cluster_id"].to_numpy()
        min_dist = dists[np.arange(n), assigned]
        df["ml_strength"] = 1.0 / (1.0 + min_dist)

    # counts per ML topic
    ml_counts = df["ml_topic"].value_counts().to_dict()
    df["ml_topic_count"] = df["ml_topic"].map(ml_counts)

    # --- 3) hybrid display topic
    has_rss_info = df["rss_topic"].str.lower().ne("uncategorized")
    use_rss = has_rss_info.mean() >= use_rss_threshold
    df["topic"] = df["rss_topic"] if use_rss else df["ml_topic"]

    return df, {"rss_labels": rss_labels, "ml_labels": ml_labels}


def short_bullet_summary(df: pd.DataFrame, topic_column: str = "topic") -> str:
    """Produce a minimal, human-readable overview by topic."""
    if df.empty:
        return "No finance news found for that date."
    lines = []
    for topic, grp in df.groupby(topic_column):
        titles = "; ".join(grp["title"].head(3).tolist())
        lines.append(f"- {topic}: {len(grp)} items (e.g., {titles})")
    return "\n".join(lines)


def gpt_daily_summary(df: pd.DataFrame, date_local: datetime, model: str) -> str:
    """Ask GPT for a super-brief daily market summary based on the DataFrame."""
    if df.empty:
        return "No finance news found for that date."
    bullets = []
    for _, row in df.head(60).iterrows():  # cap to keep prompt light
        src = row.get("source") or row.get("publisher") or ""
        bullets.append(f"- {row['title']} ({src})")
    evidence = "\n".join(bullets)

    try:
        from openai import OpenAI
    except Exception:
        return "(OpenAI SDK not installed)"

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = f"""
You are a markets editor. Based ONLY on the bullet list of headlines below,
write a concise 3–6 bullet daily wrap for {date_local.strftime('%Y-%m-%d')}.
Rules:
- Be neutral and factual, no hype.
- If headlines conflict, note uncertainty.
- Prefer macro/themes over single-company noise.
- 800 characters max.

Headlines:
{evidence}
"""
    try:
        resp = client.responses.create(model=model, input=prompt)
        try:
            return resp.output_text.strip()
        except Exception:
            # fallback for older SDKs
            return resp.output[0].content[0].text.strip()
    except Exception as e:
        return f"(GPT summary failed: {e})"


def rss_vs_ml_confusion(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tab of RSS topics vs ML topics."""
    if df.empty:
        return pd.DataFrame()
    pivot = pd.crosstab(df["rss_topic"], df["ml_topic"]).sort_index(ascending=True)
    return pivot


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
    """
    Build a daily digest from RSS feeds.

    Parameters
    ----------
    feeds : List[str]
        List of RSS/Atom feed URLs (required).
    local_tz : tzinfo (from dateutil.tz), optional
        Local timezone for date filtering; defaults to UTC if None.
    model : str, optional
        OpenAI model name for summarization; default "gpt-4o-mini".
    max_k : int, optional
        Max K for KMeans clustering; default 6.
    use_rss_threshold : float, optional
        If share of items with informative RSS categories >= this threshold,
        the hybrid display topic prefers RSS categories; default 0.4.
    """
    def __init__(
        self,
        feeds: List[str],
        local_tz=None,
        model: str = "gpt-4o-mini",
        max_k: int = 6,
        use_rss_threshold: float = 0.4
    ):
        if not feeds or not isinstance(feeds, list):
            raise ValueError("feeds (list of feed URLs) must be provided.")
        self.feeds = feeds
        self.local_tz = local_tz or tz.UTC
        self.model = model
        self.max_k = max_k
        self.use_rss_threshold = use_rss_threshold

    def build(self, date_str: Optional[str] = None) -> DigestResult:
        # resolve date (local)
        if date_str:
            target_date_local = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=self.local_tz)
        else:
            now_local = datetime.now(tz=self.local_tz)
            target_date_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)

        # fetch + frame
        entries = fetch_feed_entries(self.feeds)
        df = entries_to_dataframe(entries, target_date_local=target_date_local, local_tz=self.local_tz)
        total = len(df)

        # topics
        df, topic_maps = hybrid_topics(df, max_k=self.max_k, use_rss_threshold=self.use_rss_threshold)

        # summaries
        simple_hybrid = short_bullet_summary(df, topic_column="topic")
        simple_rss = short_bullet_summary(df, topic_column="rss_topic")
        simple_ml = short_bullet_summary(df, topic_column="ml_topic")

        # GPT summary (optional)
        gpt = None
        if os.getenv("OPENAI_API_KEY"):
            gpt = gpt_daily_summary(df, date_local=target_date_local, model=self.model)

        # confusion table
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
    ap = argparse.ArgumentParser(description="Build an economics/finance RSS daily digest.")
    ap.add_argument("--date", help="Target date in YYYY-MM-DD (local tz). Default: today.")
    ap.add_argument(
        "--feed",
        action="append",
        help="RSS/Atom feed URL (can be passed multiple times). Example: --feed https://www.yahoo.com/news/rss/finance",
    )
    ap.add_argument("--tz", default="UTC", help="IANA timezone string (e.g., America/Toronto). Default: UTC.")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name. Default: gpt-4o-mini.")
    ap.add_argument("--max-k", type=int, default=6, help="Max K for KMeans. Default: 6.")
    ap.add_argument("--use-rss-threshold", type=float, default=0.4, help="RSS prevalence threshold. Default: 0.4.")
    args = ap.parse_args()

    if not args.feed:
        raise SystemExit("Error: at least one --feed must be provided.")

    local_tz = tz.gettz(args.tz)
    agent = EconNewsAgent(
        feeds=args.feed,
        local_tz=local_tz,
        model=args.model,
        max_k=args.max_k,
        use_rss_threshold=args.use_rss_threshold,
    )
    res = agent.build(args.date)
    agent.print_overviews(res)
    agent.print_rows(res, limit=20)
    agent.print_confusion(res)


if __name__ == "__main__":
    _cli()