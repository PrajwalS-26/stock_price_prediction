"""Keyless news headline fetching for NSE stocks (Yahoo + Google News RSS)."""

import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.parse import quote

import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=IN&lang=en-IN"
GNEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

_company_cache = {}


def company_name(ticker):
    """Best-effort company name lookup (cached); falls back to base symbol."""
    if ticker in _company_cache:
        return _company_cache[ticker]

    name = None
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).get_info()
        name = info.get("longName") or info.get("shortName")
    except Exception:
        name = None

    if not name:
        name = ticker.split(".")[0].replace("^", "")
    _company_cache[ticker] = name
    return name


def fetch_headlines(ticker, limit=40):
    """Fetch recent headlines for a ticker from multiple RSS sources.

    Returns a list of {"title", "published", "source"} dicts, de-duplicated.
    """
    seen = set()
    out = []

    try:
        xml = requests.get(YAHOO_RSS.format(ticker=ticker), timeout=15, headers=UA).text
        for item in _parse_rss(xml):
            if item["title"] not in seen:
                seen.add(item["title"])
                item["source"] = "Yahoo Finance"
                out.append(item)
    except Exception:
        pass

    try:
        company = company_name(ticker)
        query = quote(f'"{company}" stock')
        xml = requests.get(GNEWS_RSS.format(query=query), timeout=15, headers=UA).text
        for item in _parse_rss(xml):
            if item["title"] not in seen:
                seen.add(item["title"])
                item["source"] = "Google News"
                out.append(item)
    except Exception:
        pass

    return out[: limit * 2]


def _parse_rss(xml_text, limit=60):
    items = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return items

    for node in root.iter("item"):
        title = node.findtext("title")
        if not title:
            continue
        published = node.findtext("pubDate")
        items.append({
            "title": title.strip(),
            "published": _parse_date(published),
        })
        if len(items) >= limit:
            break
    return items


def _parse_date(value):
    if not value:
        return None
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(value).isoformat()
    except Exception:
        return value


if __name__ == "__main__":
    for h in fetch_headlines("TCS.NS", limit=5):
        print(h["published"], "|", h["title"][:90])
