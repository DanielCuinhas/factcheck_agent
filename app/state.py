import operator
from typing import Annotated, TypedDict

from .schemas import (
    Article, AuthorCheck, ClaimAssessment, Corroboration, ExtractedItem,
    FactCheckReport, Origin, SourceReputation, Suspicion,
)


class State(TypedDict, total=False):
    # input
    url: str | None
    raw_text: str | None
    model_provider: str | None
    model: str | None
    # pipeline
    article: Article
    extracted: ExtractedItem
    origin: Origin
    reputation: SourceReputation
    authors: list[AuthorCheck]
    suspicion: Suspicion
    claims: list[ClaimAssessment]
    corroboration: Corroboration
    # parallel branches append here; the reducer merges them
    limitations: Annotated[list[str], operator.add]
    report: FactCheckReport
