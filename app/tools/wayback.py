import httpx

from ..config import settings


async def earliest_snapshot(url: str) -> dict | None:
    """Earliest 200-status Wayback Machine capture of a URL (CDX results are oldest-first)."""
    params = {
        "url": url, "output": "json", "limit": 1,
        "fl": "timestamp,original,statuscode", "filter": "statuscode:200",
    }
    try:
        async with httpx.AsyncClient(timeout=settings.http_timeout) as c:
            r = await c.get("https://web.archive.org/cdx/search/cdx", params=params)
            rows = r.json()
    except (httpx.HTTPError, ValueError):
        return None
    if len(rows) < 2:  # row 0 is the header
        return None
    ts, original, _ = rows[1]
    return {"timestamp": ts, "url": f"https://web.archive.org/web/{ts}/{original}"}
