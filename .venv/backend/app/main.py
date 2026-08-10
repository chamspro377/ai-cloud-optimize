from fastapi import FastAPI

app = FastAPI(
    title="AI Cloud Optimizer",
    version="0.1.0"
)


@app.get("/")
def root():
    return {
        "service": "AI Cloud Optimizer",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }