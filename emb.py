from typing import List
from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings

# This is the class you need.
class HuggingFaceInferenceAPIEmbeddings(Embeddings):
    """
    A custom LangChain-compatible embeddings class that uses the HuggingFace
    Inference API. It is a drop-in replacement for HuggingFaceEmbeddings
    for users who want to use the serverless API instead of a local model.
    """
    def __init__(self, api_key: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes the embedding client.

        Args:
            api_key (str): Your Hugging Face API token.
            model_name (str): The name of the sentence-transformer model on the Hub.
        """
        self.client = InferenceClient(token=api_key)
        self.model_name = model_name

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """A helper method to call the feature-extraction API."""
        # The API returns a list of embeddings, one for each text.
        return self.client.feature_extraction(texts, model=self.model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of documents using the Hugging Face Inference API.
        This method is required by the LangChain Embeddings interface.
        """
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single query using the Hugging Face Inference API.
        This method is required by the LangChain Embeddings interface.
        """
        # The API expects a list, so we wrap the single text in a list
        # and then return the first (and only) embedding from the result.
        return self._embed([text])[0]
