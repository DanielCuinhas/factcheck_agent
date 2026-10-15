"""Pydantic models: API input, intermediate data, and the final report.

The Literal enums double as the structured schema the selected model must fill in, so the model
cannot return free-form verdicts.
"""
from datetime import datetime
import json
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


# ---------- API input ----------
class CheckRequest(BaseModel):
    url: HttpUrl | None = None
    text: str | None = Field(None, max_length=50_000)
    model_provider: Literal["anthropic", "openai", "huggingface"] | None = None
    model: str | None = Field(None, min_length=1, max_length=200)

    @model_validator(mode="after")
    def _one_of(self):
        if not self.url and not (self.text and self.text.strip()):
            raise ValueError("Provide either 'url' or 'text'.")
        return self


# ---------- Intermediate ----------
class Article(BaseModel):
    url: str | None = None
    domain: str | None = None
    title: str | None = None
    text: str
    authors: list[str] = Field(default_factory=list)
    published: str | None = None
    outbound_links: list[str] = Field(default_factory=list)


class ExtractedItem(BaseModel):
    claims: list[str] = Field(
        description="Atomic, self-contained, independently checkable factual claims. No opinions."
    )
    search_queries: list[str] = Field(
        description="2-4 short queries (4-10 words) with distinctive names/numbers/phrases, in the article's language."
    )
    byline_names: list[str] = Field(description="Author names explicitly stated in the text, if any.")
    outlet_name: str | None = Field(None, description="Publisher/outlet named in the text, if any.")
    cited_sources: list[str] = Field(description="Sources the item itself cites (people, agencies, documents, posts).")
    language: str = Field(description="BCP-47 code of the article language, e.g. 'es', 'ca', 'en'.")


# ---------- Report parts ----------
class Evidence(BaseModel):
    url: str
    title: str = ""
    role: Literal["supports", "contradicts", "context", "fact_check"]
    independent: bool = Field(
        description="True if this source did its own reporting rather than repeating another source in the list."
    )
    primary: bool = Field(
        description="True for primary sources: official statements, court filings, datasets, original recordings."
    )
    note: str = ""


class ClaimAssessment(BaseModel):
    claim: str
    verdict: Literal["supported", "contradicted", "mixed", "unverified"]
    confidence: float = Field(ge=0, le=1, description="Strength of the evidence, not prior belief.")
    rationale: str = Field(description="Max ~60 words.")
    evidence: list[Evidence] = Field(default_factory=list)

    @field_validator("evidence", mode="before")
    @classmethod
    def _parse_evidence_json(cls, value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value


class Origin(BaseModel):
    earliest_found_url: str | None = None
    earliest_date: str | None = None
    platform_type: Literal[
        "news_agency", "news_outlet", "social_media", "blog_or_forum",
        "government_or_primary", "messaging_app", "unknown",
    ]
    chain: list[str] = Field(
        default_factory=list,
        description="URLs from earliest known appearance to the item under review, copied from supplied material.",
    )
    confidence: Literal["low", "medium", "high"]
    notes: str = Field(description="Include caveat that 'earliest found' may not be the true origin.")


class SourceReputation(BaseModel):
    domain: str | None = None
    rating: Literal["established", "mixed", "low_credibility", "unknown"]
    signals: list[str] = Field(description="Positive or neutral signals found.")
    flags: list[str] = Field(description="Red flags found (lookalike domain, flagged by fact-checkers, etc.).")
    domain_age_days: int | None = None
    evidence_urls: list[str] = Field(default_factory=list)


class AuthorCheck(BaseModel):
    name: str
    status: Literal["verifiable", "partially_verifiable", "no_footprint_found"]
    notes: str
    evidence_urls: list[str] = Field(default_factory=list)


class AuthorChecks(BaseModel):
    authors: list[AuthorCheck]


class Suspicion(BaseModel):
    score: float = Field(ge=0, le=1)
    level: Literal["low", "medium", "high"]
    indicators: list[str] = Field(description="Specific suspicious signals, quoting short snippets where useful.")
    mitigating_factors: list[str]


class Corroboration(BaseModel):
    independent_sources: int
    primary_sources: int
    summary: str


class OverallAssessment(BaseModel):
    verdict: Literal["likely_reliable", "mixed_or_unclear", "likely_misleading", "insufficient_evidence"]
    summary: str = Field(description="Max ~120 words. State what is and is not established.")


class FactCheckReport(BaseModel):
    input_type: Literal["url", "text"]
    url: str | None
    claims: list[ClaimAssessment]
    origin: Origin
    source_reputation: SourceReputation
    authors: list[AuthorCheck]
    suspicion: Suspicion
    corroboration: Corroboration
    overall: OverallAssessment
    limitations: list[str]
    generated_at: datetime
