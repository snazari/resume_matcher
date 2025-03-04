from typing import List, Optional
from sentence_transformers import SentenceTransformer
import torch
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, hf_config, batch_size: int = 30):
        self.hf_config = hf_config
        self.batch_size = batch_size

        if hf_config.use_local:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer(hf_config.model_id, device=device)
            logger.info(f"Loaded local model '{hf_config.model_id}' on device: {device}")
        else:
            self.api_url = hf_config.api_url
            self.headers = {"Authorization": f"Bearer {hf_config.api_token}"}
            self.model = None  # No local model when using API

    def generate_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        if self.hf_config.use_local:
            if self.model is None:
                raise ValueError("Local model is not initialized.")
            try:
                embeddings = self.model.encode(
                    texts,
                    batch_size=self.batch_size,
                    convert_to_tensor=True
                )
                return embeddings.cpu().numpy().tolist()
            except Exception as e:
                logger.error(f"Error generating embeddings with local model: {str(e)}")
                return None
        else:
            if not hasattr(self, 'api_url') or not hasattr(self, 'headers'):
                raise ValueError("API configuration is not set.")
            # Add API request logic here if needed (not relevant for your local setup)
            logger.error("API mode not implemented in this example.")
            return None