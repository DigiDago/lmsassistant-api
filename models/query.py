from pydantic import BaseModel

class Query(BaseModel):
    token: str
    courseid: int
    message: str
    history: list[str] = []
    max_call_per_day: int
    model_name: str
    llm_provider: str
    doc_language: str
    instruction: str