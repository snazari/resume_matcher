"""
Simplified module for storing and retrieving embeddings with debug output.
"""
import os
import pickle
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)


class SimpleEmbeddingStorage:
    """Simplified class for storing and retrieving embeddings with debug output."""

    def __init__(self, storage_dir: Union[str, Path]):
        """
        Initialize the embedding storage.

        Args:
            storage_dir: Directory to store embedding files
        """
        self.storage_dir = Path(storage_dir)
        os.makedirs(self.storage_dir, exist_ok=True)

        # Create subdirectories
        self.candidates_dir = self.storage_dir / "candidates"
        self.jobs_dir = self.storage_dir / "jobs"
        os.makedirs(self.candidates_dir, exist_ok=True)
        os.makedirs(self.jobs_dir, exist_ok=True)

        logger.info(f"Initialized embedding storage at {self.storage_dir}")

    def _get_safe_filename(self, text_id: str) -> str:
        """Create a safe filename from an identifier."""
        # Remove any non-alphanumeric characters and replace with underscore
        safe_id = "".join(c if c.isalnum() else "_" for c in text_id)
        return f"{safe_id}.pkl"

    def store_embeddings(self, df: pd.DataFrame,
                         id_column: str,
                         embedding_column: str,
                         is_candidate: bool = True) -> None:
        """Store embeddings for either candidates or jobs."""
        entity_type = "candidate" if is_candidate else "job"
        storage_dir = self.candidates_dir if is_candidate else self.jobs_dir
        stored_count = 0

        for idx, row in df.iterrows():
            # Skip if no embedding
            if row[embedding_column] is None:
                print("Skipping Row...")
                continue

            text_id = str(row[id_column]).strip()
            if not text_id:
                continue

            # Create filename and path
            filename = self._get_safe_filename(text_id)
            filepath = storage_dir / filename

            # Store embedding
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(row[embedding_column], f)
                stored_count += 1
            except Exception as e:
                logger.error(f"Error storing {entity_type} embedding for {text_id}: {str(e)}")

        logger.info(f"Stored {stored_count} {entity_type} embeddings")

    def load_embeddings(self, df: pd.DataFrame,
                        id_column: str,
                        embedding_column: str,
                        is_candidate: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load stored embeddings for a DataFrame.

        Returns:
            Tuple of (DataFrame with embeddings loaded, mask of rows needing embeddings)
        """
        entity_type = "candidate" if is_candidate else "job"
        storage_dir = self.candidates_dir if is_candidate else self.jobs_dir

        # Ensure embedding column exists
        result_df = df.copy()
        if embedding_column not in result_df.columns:
            result_df[embedding_column] = None

        # Track which entities need embeddings
        needs_embedding = np.ones(len(result_df), dtype=bool)
        loaded_count = 0

        # Try to load existing embeddings
        for i, (idx, row) in enumerate(result_df.iterrows()):
            text_id = str(row[id_column]).strip()
            if not text_id:
                continue

            # Check if embedding exists
            filename = self._get_safe_filename(text_id)
            filepath = storage_dir / filename

            if filepath.exists():
                try:
                    with open(filepath, 'rb') as f:
                        embedding = pickle.load(f)
                    result_df.at[idx, embedding_column] = embedding
                    needs_embedding[i] = False
                    loaded_count += 1
                    logger.debug(f"Loaded {entity_type} embedding for {text_id}")
                except Exception as e:
                    logger.error(f"Error loading {entity_type} embedding for {text_id}: {str(e)}")

        logger.info(f"Loaded {loaded_count} stored {entity_type} embeddings, "
                    f"{sum(needs_embedding)} need to be generated")

        # Create mask
        embedding_mask = pd.Series(needs_embedding, index=result_df.index)

        return result_df, embedding_mask