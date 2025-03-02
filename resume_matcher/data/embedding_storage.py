"""
Module for storing and retrieving embeddings.
"""
import os
import json
import pickle
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)


class EmbeddingStorage:
    """Class for storing and retrieving embeddings."""

    def __init__(self, storage_dir: Union[str, Path]):
        """
        Initialize the embedding storage.

        Args:
            storage_dir: Directory to store embedding files
        """
        self.storage_dir = Path(storage_dir)
        os.makedirs(self.storage_dir, exist_ok=True)
        self.candidates_index = {}  # Maps candidate name or ID to file path
        self.jobs_index = {}  # Maps job ID or title to file path
        self._load_index()

    def _load_index(self):
        """Load the embedding indexes if they exist."""
        candidate_index_path = self.storage_dir / "candidate_index.json"
        jobs_index_path = self.storage_dir / "jobs_index.json"

        if candidate_index_path.exists():
            try:
                with open(candidate_index_path, 'r') as f:
                    self.candidates_index = json.load(f)
                logger.info(f"Loaded index for {len(self.candidates_index)} candidates")
            except Exception as e:
                logger.error(f"Error loading candidate index: {str(e)}")

        if jobs_index_path.exists():
            try:
                with open(jobs_index_path, 'r') as f:
                    self.jobs_index = json.load(f)
                logger.info(f"Loaded index for {len(self.jobs_index)} jobs")
            except Exception as e:
                logger.error(f"Error loading jobs index: {str(e)}")

    def _save_index(self):
        """Save the embedding indexes."""
        try:
            with open(self.storage_dir / "candidate_index.json", 'w') as f:
                json.dump(self.candidates_index, f)

            with open(self.storage_dir / "jobs_index.json", 'w') as f:
                json.dump(self.jobs_index, f)

            logger.info("Saved embedding indexes")
        except Exception as e:
            logger.error(f"Error saving embedding indexes: {str(e)}")

    def store_candidate_embeddings(self, candidates_df: pd.DataFrame,
                                   id_column: str = 'Name ',
                                   embedding_column: str = 'hf_embedding'):
        """
        Store candidate embeddings.

        Args:
            candidates_df: DataFrame with candidate embeddings
            id_column: Column name for candidate identifier
            embedding_column: Column name for embeddings
        """
        candidates_dir = self.storage_dir / "candidates"
        os.makedirs(candidates_dir, exist_ok=True)

        # Create a copy with just the necessary columns
        to_store = candidates_df[[id_column, embedding_column]].copy()

        # Process each candidate
        for idx, row in to_store.iterrows():
            candidate_id = str(row[id_column]).strip()
            embedding = row[embedding_column]

            if not candidate_id:
                logger.warning(f"Empty candidate ID at index {idx}, skipping")
                continue

            # Create a safe filename
            safe_id = "".join(c if c.isalnum() else "_" for c in candidate_id)
            filename = f"{safe_id}.pkl"
            filepath = candidates_dir / filename

            # Store the embedding
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(embedding, f)

                # Update the index
                self.candidates_index[candidate_id] = str(filepath.relative_to(self.storage_dir))
                logger.debug(f"Stored embedding for candidate: {candidate_id}")
            except Exception as e:
                logger.error(f"Error storing embedding for candidate {candidate_id}: {str(e)}")

        # Save the updated index
        self._save_index()
        logger.info(f"Stored embeddings for {len(to_store)} candidates")

    def store_job_embeddings(self, jobs_df: pd.DataFrame,
                             id_column: str = 'Role',
                             embedding_column: str = 'hf_embedding'):
        """
        Store job embeddings.

        Args:
            jobs_df: DataFrame with job embeddings
            id_column: Column name for job identifier
            embedding_column: Column name for embeddings
        """
        jobs_dir = self.storage_dir / "jobs"
        os.makedirs(jobs_dir, exist_ok=True)

        # Create a copy with just the necessary columns
        to_store = jobs_df[[id_column, embedding_column]].copy()

        # Process each job
        for idx, row in to_store.iterrows():
            job_id = str(row[id_column]).strip()
            embedding = row[embedding_column]

            if not job_id:
                logger.warning(f"Empty job ID at index {idx}, skipping")
                continue

            # Create a safe filename
            safe_id = "".join(c if c.isalnum() else "_" for c in job_id)
            filename = f"{safe_id}.pkl"
            filepath = jobs_dir / filename

            # Store the embedding
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(embedding, f)

                # Update the index
                self.jobs_index[job_id] = str(filepath.relative_to(self.storage_dir))
                logger.debug(f"Stored embedding for job: {job_id}")
            except Exception as e:
                logger.error(f"Error storing embedding for job {job_id}: {str(e)}")

        # Save the updated index
        self._save_index()
        logger.info(f"Stored embeddings for {len(to_store)} jobs")

    def get_candidate_embedding(self, candidate_id: str) -> Optional[List[float]]:
        """
        Retrieve a candidate's embedding.

        Args:
            candidate_id: Candidate identifier

        Returns:
            Embedding vector or None if not found
        """
        if candidate_id not in self.candidates_index:
            logger.debug(f"No stored embedding found for candidate: {candidate_id}")
            return None

        try:
            filepath = self.storage_dir / self.candidates_index[candidate_id]
            with open(filepath, 'rb') as f:
                embedding = pickle.load(f)
            return embedding
        except Exception as e:
            logger.error(f"Error loading embedding for candidate {candidate_id}: {str(e)}")
            return None

    def get_job_embedding(self, job_id: str) -> Optional[List[float]]:
        """
        Retrieve a job's embedding.

        Args:
            job_id: Job identifier

        Returns:
            Embedding vector or None if not found
        """
        if job_id not in self.jobs_index:
            logger.debug(f"No stored embedding found for job: {job_id}")
            return None

        try:
            filepath = self.storage_dir / self.jobs_index[job_id]
            with open(filepath, 'rb') as f:
                embedding = pickle.load(f)
            return embedding
        except Exception as e:
            logger.error(f"Error loading embedding for job {job_id}: {str(e)}")
            return None

    def load_candidate_embeddings(self, candidates_df: pd.DataFrame,
                                  id_column: str = 'Name ',
                                  embedding_column: str = 'hf_embedding'):
        """
        Load stored embeddings for candidates.

        Args:
            candidates_df: DataFrame with candidate information
            id_column: Column name for candidate identifier
            embedding_column: Column name for embeddings

        Returns:
            DataFrame with embeddings added or updated and mask of rows needing embeddings
        """
        result_df = candidates_df.copy()

        # Ensure embedding column exists
        if embedding_column not in result_df.columns:
            result_df[embedding_column] = None

        # Track which candidates need new embeddings
        candidates_to_embed = []

        # Try to load stored embeddings
        for idx, row in result_df.iterrows():
            candidate_id = str(row[id_column]).strip()

            if not candidate_id:
                logger.warning(f"Empty candidate ID at index {idx}, skipping")
                continue

            # Try to get stored embedding
            embedding = self.get_candidate_embedding(candidate_id)

            if embedding is not None:
                result_df.at[idx, embedding_column] = embedding
            else:
                candidates_to_embed.append(idx)

        loaded_count = len(result_df) - len(candidates_to_embed)
        logger.info(
            f"Loaded {loaded_count} stored candidate embeddings, {len(candidates_to_embed)} need to be generated")

        # Create a mask for candidates that need new embeddings
        embedding_mask = result_df.index.isin(candidates_to_embed)

        return result_df, embedding_mask

    def load_job_embeddings(self, jobs_df: pd.DataFrame,
                            id_column: str = 'Role',
                            embedding_column: str = 'hf_embedding'):
        """
        Load stored embeddings for jobs.

        Args:
            jobs_df: DataFrame with job information
            id_column: Column name for job identifier
            embedding_column: Column name for embeddings

        Returns:
            DataFrame with embeddings added or updated and mask of rows needing embeddings
        """
        result_df = jobs_df.copy()

        # Ensure embedding column exists
        if embedding_column not in result_df.columns:
            result_df[embedding_column] = None
    
        # Track which jobs need new embeddings
        jobs_to_embed = []

        # Try to load stored embeddings
        for idx, row in result_df.iterrows():
            job_id = str(row[id_column]).strip()

            if not job_id:
                logger.warning(f"Empty job ID at index {idx}, skipping")
                continue

            # Try to get stored embedding
            embedding = self.get_job_embedding(job_id)

            if embedding is not None:
                result_df.at[idx, embedding_column] = embedding
            else:
                jobs_to_embed.append(idx)

        loaded_count = len(result_df) - len(jobs_to_embed)
        logger.info(f"Loaded {loaded_count} stored job embeddings, {len(jobs_to_embed)} need to be generated")

        # Create a mask for jobs that need new embeddings
        embedding_mask = result_df.index.isin(jobs_to_embed)

        return result_df, embedding_mask