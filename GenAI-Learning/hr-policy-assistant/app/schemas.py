
from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class SourceItem(BaseModel):
    document: str
    page: int
    excerpt: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
