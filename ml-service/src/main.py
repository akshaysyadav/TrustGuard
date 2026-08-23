from fastapi import FastAPI

from src.api.phishing_routes import router as phishing_router
from src.api.fake_review_routes import router as fake_review_router


app = FastAPI(
    title="TrustGuard ML Service",
    description="Machine Learning API for TrustGuard",
    version="1.0.0"
)


app.include_router(phishing_router)
app.include_router(fake_review_router)


@app.get("/")
def root():
    return {
        "message": "TrustGuard ML Service is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }