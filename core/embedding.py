import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class Embedding:
    def __init__(self, embedding_config: dict):
        self.model_name: str = embedding_config["model_name"]
        self.model: SentenceTransformer | None = None

    def load_model(self) -> None:
        """Load the SentenceTransformer model into memory."""
        self.model = SentenceTransformer(self.model_name)

    def get_embedding(self, text: str) -> np.ndarray:
        """Encode a single text string and return its embedding vector."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model.encode(text)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a list of text strings and return the embedding matrix."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model.encode(texts, show_progress_bar=True)