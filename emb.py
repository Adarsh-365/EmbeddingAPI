from sentence_transformers import SentenceTransformer

# 1. Load a pre-trained Sentence Transformer model
# "all-MiniLM-L6-v2" is a good and fast general-purpose model.
model = SentenceTransformer("all-MiniLM-L6-v2")



def get_embedding(text):
    """Generate embedding for a single text input."""
    return model.encode([text])[0]