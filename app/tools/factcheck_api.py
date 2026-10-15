"""Google Fact Check Tools API (aggregates ClaimReview markup from fact-checkers)."""
import httpx
from pydantic import BaseModel

from ..config import settings


class FactCheckHit(BaseModel):
    claim: str
    claimant: str | None = None
    publisher: str = ""
    url: str = ""
    rating: str = ""
    review_date: str | None = None


async def search_claims(query: str, language: str | None = None) -> list[FactCheckHit]:
    if not settings.google_factcheck_api_key:
        return []
    params = {"query": query[:200], "key": settings.google_factcheck_api_key, "pageSize": 5}
    if language:
        params["languageCode"] = language
    try:
        async with httpx.AsyncClient(timeout=settings.http_timeout) as c:
            r = await c.get("https://factchecktools.googleapis.com/v1alpha1/claims:search", params=params)
            r.raise_for_status()
            data = r.json()
    except (httpx.HTTPError, ValueError):
        return []
    hits = []
    for c in data.get("claims", []):
        for rv in c.get("claimReview", []):
            hits.append(
                FactCheckHit(
                    claim=c.get("text", ""),
                    claimant=c.get("claimant"),
                    publisher=rv.get("publisher", {}).get("name", ""),
                    url=rv.get("url", ""),
                    rating=rv.get("textualRating", ""),
                    review_date=rv.get("reviewDate"),
                )
            )
    return hits
