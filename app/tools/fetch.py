"""Safe page fetching + article extraction.

The service fetches user-supplied URLs, so every hop (including redirects) is
checked to resolve to a public IP address (basic SSRF protection).
"""
import asyncio
import ipaddress
import json
import socket
from urllib.parse import urljoin, urlparse

import httpx
import tldextract
import trafilatura
from bs4 import BeautifulSoup

from ..config import settings
from ..schemas import Article

_tld = tldextract.TLDExtract(suffix_list_urls=())  # bundled suffix list, no network
UA = "Mozilla/5.0 (compatible; FactCheckAgent/0.1)"
MAX_BYTES = 2_000_000


class FetchError(Exception):
    pass


def registered_domain(url_or_host: str) -> str | None:
    host = urlparse(url_or_host).hostname if "://" in url_or_host else url_or_host
    if not host:
        return None
    ext = _tld(host)
    return f"{ext.domain}.{ext.suffix}" if ext.suffix else (ext.domain or None)


def _check_host_sync(host: str) -> None:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise FetchError(f"Cannot resolve host {host}") from e
    for info in infos:
        if not ipaddress.ip_address(info[4][0]).is_global:
            raise FetchError("Refusing to fetch a non-public address")


async def safe_get(url: str, max_redirects: int = 5) -> httpx.Response:
    async with httpx.AsyncClient(
        timeout=settings.http_timeout, headers={"User-Agent": UA}, follow_redirects=False
    ) as client:
        for _ in range(max_redirects + 1):
            p = urlparse(url)
            if p.scheme not in ("http", "https") or not p.hostname:
                raise FetchError("Invalid URL")
            await asyncio.to_thread(_check_host_sync, p.hostname)
            try:
                r = await client.get(url)
            except httpx.HTTPError as e:
                raise FetchError(f"Request failed: {e}") from e
            if r.is_redirect and (loc := r.headers.get("location")):
                url = urljoin(str(r.url), loc)
                continue
            return r
    raise FetchError("Too many redirects")


def _meta(soup: BeautifulSoup, *names: str) -> list[str]:
    out = []
    for n in names:
        for tag in soup.find_all("meta", attrs={"name": n}) + soup.find_all("meta", attrs={"property": n}):
            v = (tag.get("content") or "").strip()
            if v and not v.startswith("http"):
                out.append(v)
    return out


async def fetch_article(url: str) -> Article:
    r = await safe_get(url)
    if r.status_code >= 400:
        raise FetchError(f"HTTP {r.status_code} fetching {url}")
    html = r.content[:MAX_BYTES].decode(r.encoding or "utf-8", errors="ignore")

    extracted = trafilatura.extract(
        html, url=url, output_format="json", with_metadata=True, include_comments=False
    )
    meta = json.loads(extracted) if extracted else {}
    text = (meta.get("text") or "").strip()
    if len(text) < 200:
        raise FetchError("Could not extract article text (paywall, JS-rendered page, or blocked).")

    soup = BeautifulSoup(html, "lxml")
    raw_authors = [meta.get("author") or ""] + _meta(soup, "author", "article:author", "byl")
    authors: list[str] = []
    for chunk in raw_authors:
        for name in chunk.replace(" and ", ";").replace(",", ";").split(";"):
            name = name.strip()
            if name and name not in authors:
                authors.append(name)

    own = registered_domain(url)
    links: list[str] = []
    for a in soup.find_all("a", href=True):
        href = urljoin(url, a["href"])
        if href.startswith(("http://", "https://")) and registered_domain(href) != own and href not in links:
            links.append(href)

    published = meta.get("date") or next(iter(_meta(soup, "article:published_time", "date")), None)
    return Article(
        url=url, domain=own, title=meta.get("title"), text=text,
        authors=authors, published=published, outbound_links=links[:40],
    )


async def probe_pages(host: str) -> dict[str, int | None]:
    """Weak legitimacy signal: does the site have about/contact/corrections pages?"""
    paths = ["/about", "/about-us", "/contact", "/corrections", "/ethics", "/editorial-policy", "/masthead"]

    async def one(p: str):
        try:
            return p, (await safe_get(f"https://{host}{p}")).status_code
        except FetchError:
            return p, None

    return dict(await asyncio.gather(*(one(p) for p in paths)))
