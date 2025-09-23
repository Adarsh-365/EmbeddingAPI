from fastapi import FastAPI
from emb import HuggingFaceInferenceAPIEmbeddings
app = FastAPI()
import os
HF_API_TOKEN = os.environ.get("HF_API_TOKEN") 




@app.get("/")
async def read_root():
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_API_TOKEN,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return {"message": "Hello, FastAPI!","model":embeddings}

@app.get("/rest/{TEXT}")
async def read_item(TEXT: str):
    embedding = get_embedding(TEXT)
    return {"TEXT": TEXT, "Embeddings": embedding.tolist()}
    
