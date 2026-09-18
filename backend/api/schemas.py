from typing import Literal

from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    text: str


AttributeType = Literal["AGE", "GENDER", "LOCATION", "HEALTH", "FACILITY", "DATE", "OCCUPATION", "FINANCIAL", "FAMILY", "PHONE_NUMBER", "GOVERNMENT_ID"]

class Message(BaseModel):
    id: int
    text: str


class DetectedAttribute(BaseModel):
    type: AttributeType
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    specificity: float = Field(ge=0.0, le=1.0)


# Keep these values aligned with docs/api-contract.md.
class ConversationAnalysis(BaseModel):
    conversation_id: str
    messages: list[Message] = Field(default_factory=list)
    attributes: list[DetectedAttribute]
    # Exact values: village_level_location, exact_age, exact_date,
    # sensitive_health_attribute, uniqueness_word.
    contextual_indicators: list[str]
    uniqueness_terms: list[str]
