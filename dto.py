from typing import Optional, List

from pydantic import BaseModel


class SearchRequest(BaseModel):
    user_query: str
    place_description: str


class RelevanceRequest(BaseModel):
    user_query: str
    place_description: str
    use_agent: Optional[bool] = True
    use_search: Optional[bool] = False
    use_bert: Optional[bool] = False
    bert_threshold: Optional[float] = 0.5


class RelevanceResponse(BaseModel):
    decision: Optional[int] = None
    bert_score: Optional[float] = None
    agent_output: Optional[str] = None
    used: List[str] = []
    note: Optional[str] = None
