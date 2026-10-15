from datetime import datetime, timezone

import httpx

from ..config import settings


async def domain_age_days(domain: str) -> int | None:
    """Domain age from RDAP registration event. None if the registry doesn't answer."""
    try:
        async with httpx.AsyncClient(timeout=settings.http_timeout, follow_redirects=True) as c:
            r = await c.get(f"https://rdap.org/domain/{domain}")
            r.raise_for_status()
            events = r.json().get("events", [])
    except (httpx.HTTPError, ValueError):
        return None
    for ev in events:
        if ev.get("eventAction") == "registration":
            try:
                dt = datetime.fromisoformat(ev["eventDate"].replace("Z", "+00:00"))
            except (KeyError, ValueError):
                return None
            return (datetime.now(timezone.utc) - dt).days
    return None
