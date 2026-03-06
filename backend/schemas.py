from pydantic import BaseModel

# Request schema
class AnxietyRequest(BaseModel):
    text: str

# Response schema
class AnxietyResponse(BaseModel):
    anxiety_level: int
    anxiety_category: str