from fastapi import FastAPI


app = FastAPI(
    title="Clause API",
    description="Clause Shop RAG assistant API",
    version="0.1.0",
)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Welcome to Clause API", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/hello")
def hello(name: str = "World") -> dict[str, str]:
    return {"message": f"Hello, {name}!"}
