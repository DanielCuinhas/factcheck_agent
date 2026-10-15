"""Web search via Tavily. Swap this module to use another provider."""
import logging

import httpx
from pydantic import BaseModel

from ..config import settings

log = logging.getLogger(__name__)


class SearchResult(BaseModel):
    url: str
    title: str = ""
    snippet: str = ""
    published: str | None = None


async def web_search(query: str, *, max_results: int = 6, news: bool = False) -> list[SearchResult]:
    if not settings.tavily_api_key:
        return []
    payload = {
        "query": query,
        "max_results": max_results,
        "topic": "news" if news else "general",
        "search_depth": "basic",
    }
    try:
        async with httpx.AsyncClient(timeout=settings.http_timeout) as c:
            r = await c.post(
                "https://api.tavily.com/search",
                json=payload,
                headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
            )
            r.raise_for_status()
    except httpx.HTTPError as e:
        log.warning("search failed for %r: %s", query, e)
        return []
    return [
        SearchResult(
            url=x["url"],
            title=x.get("title", ""),
            snippet=(x.get("content") or "")[:500],
            published=x.get("published_date"),
        )
        for x in r.json().get("results", [])
    ]
