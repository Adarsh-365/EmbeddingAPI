from fastapi import FastAPI
from emb import get_embedding
app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello, FastAPI!"}

@app.get("/rest/{TEXT}")
async def read_item(TEXT: str):
    embedding = get_embedding(TEXT)
    return {"TEXT": TEXT, "Embeddings": embedding.tolist()}
    