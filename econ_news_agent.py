"""
econ_news_agent.py — RSS-based daily economics/finance digest with hybrid topics and GPT summary.

No globals are hard-coded. Pass everything in via run():
    run(date_str, feeds, local_tz, model, use_rss_threshold=0.4, max_k=6)
Returns:
    (df: pandas.DataFrame, gpt_summary: Optional[str])

Exported symbols (for `from econ_news_agent import *`):
    run, normalize_datetime, same_calendar_day, fetch_feed_entries,
    entries_to_dataframe, hybrid_topics, short_bullet_summary,
    gpt_daily_summary, rss_vs_ml_confusion
"""

import os
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime

import numpy as np
import pandas as pd
import feedparser
from dateutil import tz, parser as dateparser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

__all__ = [
    "run",
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
    # Convenience columns
    df["published_local"] = df["published_dt_utc"].apply(lambda d: d.astimezone(local_tz) if pd.notnull(d) else None)
    df["text_for_cluster"] = (df["title"].fillna("") + " — " + df["summary"].fillna("")).str.strip()
    df["source"] = df["link"].apply(lambda u: u.split("/")[2] if isinstance(u, str) and "://" in u else "")
    df["categories"] = df["tags"].apply(lambda ts: ", ".join(ts) if isinstance(ts, list) and ts else "")
    cols = ["published_local", "title", "link", "publisher", "source", "categories",
            "summary", "tags", "published_dt_utc", "text_for_cluster"]
    df = df[cols]
    df = df.sort_values("published_local")
    return df


def hybrid_topics(df: pd.DataFrame, max_k: int = 6, use_rss_threshold: float = 0.4) -> Tuple[pd.DataFrame, Dict[str, Dict[int, str]]]:
    """
    Keep RSS categories and also compute KMeans clusters on TF-IDF.

    Adds columns:
      - rss_topic, rss_topic_count
      - ml_cluster_id, ml_topic, ml_topic_count, ml_strength
      - topic  (hybrid display label: prefers RSS if prevalent)
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


def gpt_daily_summary(df: pd.DataFrame, date_local: datetime, model: str) -> Optional[str]:
    """Ask GPT for a super-brief daily market summary based on the DataFrame."""
    if df.empty:
        return "No finance news found for that date."

    bullets = []
    for _, row in df.head(60).iterrows():  # cap to keep the prompt light
        src = row.get("source") or row.get("publisher") or ""
        bullets.append(f"- {row['title']} ({src})")
    evidence = "\n".join(bullets)

    try:
        from openai import OpenAI
    except Exception:
        return "(OpenAI SDK not installed)"

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "(OPENAI_API_KEY not set)"

    client = OpenAI(api_key=api_key)

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
        # Latest SDK: resp.output_text; older SDKs: nested list
        return getattr(resp, "output_text", None) or resp.output[0].content[0].text
    except Exception as e:
        return f"(GPT summary failed: {e})"


def rss_vs_ml_confusion(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tab of RSS topics vs ML topics."""
    if df.empty:
        return pd.DataFrame()
    pivot = pd.crosstab(df["rss_topic"], df["ml_topic"]).sort_index(ascending=True)
    return pivot


# ----------------- MAIN ENTRY (NO CLASS) -----------------

def run(
    date_str: Optional[str],
    feeds: List[str],
    local_tz=None,
    model: str = "gpt-4o-mini",
    use_rss_threshold: float = 0.4,
    max_k: int = 6,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Build the digest and return (df, gpt_summary).

    Parameters
    ----------
    date_str : 'YYYY-MM-DD' or None    -> if None, uses today in local_tz
    feeds    : list of RSS/Atom URLs   -> required
    local_tz : tzinfo (dateutil.tz)    -> defaults to UTC if None
    model    : str                      -> OpenAI model for summary
    use_rss_threshold : float           -> prevalence threshold for RSS topics
    max_k   : int                      -> max clusters for KMeans

    Returns
    -------
    df : pandas.DataFrame
    gpt_summary : Optional[str]
    """
    if not feeds or not isinstance(feeds, list):
        raise ValueError("feeds (list of feed URLs) must be provided.")
    local_tz = local_tz or tz.UTC

    # Resolve date
    if date_str:
        target_date_local = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=local_tz)
    else:
        now_local = datetime.now(tz=local_tz)
        target_date_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)

    # Fetch + frame
    entries = fetch_feed_entries(feeds)
    df = entries_to_dataframe(entries, target_date_local=target_date_local, local_tz=local_tz)

    # Topics
    df, _ = hybrid_topics(df, max_k=max_k, use_rss_threshold=use_rss_threshold)

    # GPT summary
    gpt = gpt_daily_summary(df, date_local=target_date_local, model=model)

    return df, gpt

if __name__ == "__main__":
    _cli()