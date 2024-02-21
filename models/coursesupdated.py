from pydantic import BaseModel

class CoursesUpdated(BaseModel):
    token: str