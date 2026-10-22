from typing import TypedDict, Annotated

from pydantic import BaseModel, Field
import operator

class SearchResult(BaseModel):
    title: str
    url: str
    content: str
    score: float | None = None

class Evidence(BaseModel):
    the_id : int = 0
    source_title: str
    source_url: str
    excerpt: str

class PlanResearch(BaseModel):
    sub_questions: list[str] = Field(description="需要研究的子问题列表")

class ResearchState(TypedDict):
    question: str
    search_query: Annotated[list[str], operator.add]
    search_results: list[SearchResult]
    evidence: Annotated[list[Evidence], operator.add]
    errors: list[str]
    turn: int = 2
    search_plan: list[str]
    query_generation_failed: bool
    report: str
    citation_errors: list
    review_approved: bool
    review_feedback: str
    missing_topics: list[str]
    review_round: int
    max_review_round: int = 2

class LocalDocument(BaseModel):
    source_path: str
    content: str

class ReviewResult(BaseModel):
    approved: bool = Field(description="报告质量是否通过")
    feedback: str = Field(description="反馈内容")
    missing_topics: list[str] = Field(description="还缺少的话题")