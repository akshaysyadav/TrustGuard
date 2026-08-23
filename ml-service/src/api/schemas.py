from pydantic import BaseModel, Field


class URLRequest(BaseModel):
    url: str = Field(
        ...,
        min_length=1,
        description="URL to analyze for phishing"
    )


class URLPredictionResponse(BaseModel):
    verdict: str
    confidence: float
    reason: str

class ReviewRequest(BaseModel):
    review: str = Field(
        ...,
        min_length=1,
        description="Review text to analyze for fake review detection"
    )


class ReviewPredictionResponse(BaseModel):
    verdict: str
    reason: str