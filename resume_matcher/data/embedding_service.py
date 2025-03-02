"""
Service for generating text embeddings using Hugging Face API.
"""
import requests
import time
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import backoff  # You'll need to add this dependency

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings from text using Hugging Face inference API."""
    
    def __init__(self, api_url: str, api_token: str, batch_size: int = 30, max_retries: int = 3):
        """
        Initialize the embedding service.
        
        Args:
            api_url: URL for the Hugging Face inference API
            api_token: API token for authentication
            batch_size: Maximum number of texts to embed in a single batch
            max_retries: Maximum number of retries for failed API calls
        """
        self.api_url = api_url
        self.headers = {"Authorization": f"Bearer {api_token}"}
        self.batch_size = batch_size
        self.max_retries = max_retries
    
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, requests.exceptions.HTTPError),
        max_tries=3,
        giveup=lambda e: isinstance(e, requests.exceptions.HTTPError) and e.response.status_code in [400, 401, 403]
    )
    def _query_api(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Send a request to the Hugging Face Inference API.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors or None if the request failed
        """
        try:
            response = requests.post(
                self.api_url, 
                headers=self.headers, 
                json={"inputs": texts, "options": {"wait_for_model": True}}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in API request: {str(e)}")
            return None
    
    def generate_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Generate embeddings for a list of texts, automatically batching if necessary.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors or None if the request failed
        """
        if not texts:
            logger.warning("No texts provided for embedding generation")
            return []
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            logger.info(f"Processing embedding batch {i // self.batch_size + 1} of {(len(texts) - 1) // self.batch_size + 1}")
            
            batch_embeddings = self._query_api(batch_texts)
            
            if batch_embeddings is None:
                logger.error(f"Failed to get embeddings for batch {i // self.batch_size + 1}")
                return None
            
            # Validate the response format
            if not isinstance(batch_embeddings, list) or not all(isinstance(item, list) for item in batch_embeddings):
                logger.error(f"Invalid response format received for batch {i // self.batch_size + 1}")
                return None
            
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings

    def embed_dataframe(
            self,
            df: pd.DataFrame,
            text_column: str,
            embedding_column: str = 'hf_embedding',
            mask: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Add embeddings to a DataFrame.

        Args:
            df: pandas DataFrame containing the texts
            text_column: Name of the column containing the texts to embed
            embedding_column: Name of the column to store the embeddings
            mask: Boolean mask indicating which rows need new embeddings (True = needs embedding)

        Returns:
            DataFrame with embeddings added
        """
        # Make sure the embedding_column exists
        if embedding_column not in df.columns:
            df[embedding_column] = None

        # If no mask provided, embed all rows
        if mask is None:
            mask = pd.Series(True, index=df.index)

        # Get only the texts that need embedding
        texts_to_embed = df.loc[mask, text_column].tolist()

        if not texts_to_embed:
            logger.info("No texts need embedding")
            return df  # All embeddings already loaded

        logger.info(f"Generating embeddings for {len(texts_to_embed)} items")
        embeddings = self.generate_embeddings(texts_to_embed)

        if embeddings is None:
            logger.error("Failed to generate embeddings")
            return df  # Return original DataFrame without embeddings

        # Update only the rows that needed embedding
        masked_indices = df.loc[mask].index
        for i, idx in enumerate(masked_indices):
            df.at[idx, embedding_column] = embeddings[i]

        return df