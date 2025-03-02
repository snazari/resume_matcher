"""
Module for vector database operations in the Resume Matcher system.
Provides efficient storage and retrieval of embeddings using FAISS.
"""
import os
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import faiss
import json

logger = logging.getLogger(__name__)


class VectorDatabase:
    """Vector database for efficient storage and retrieval of embeddings."""

    def __init__(self, storage_dir: Union[str, Path]):
        """
        Initialize the vector database.

        Args:
            storage_dir: Directory to store the vector database files
        """
        self.storage_dir = Path(storage_dir)
        self.candidates_dir = self.storage_dir / "candidates"
        self.jobs_dir = self.storage_dir / "jobs"

        # Create directories if they don't exist
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.candidates_dir, exist_ok=True)
        os.makedirs(self.jobs_dir, exist_ok=True)

        # Initialize indexes
        self.candidate_index = None
        self.job_index = None

        # Initialize id mapping
        self.candidate_id_map = {}  # Maps index position to candidate ID
        self.candidate_id_reverse_map = {}  # Maps candidate ID to index position
        self.job_id_map = {}  # Maps index position to job ID
        self.job_id_reverse_map = {}  # Maps job ID to index position

        # Load existing indexes if they exist
        self._load_indexes()

        logger.info(f"Initialized vector database at {self.storage_dir}")

    def _load_indexes(self) -> None:
        """Load existing FAISS indexes and ID mappings if they exist."""
        # Load candidate index
        candidate_index_path = self.candidates_dir / "faiss_index.bin"
        candidate_map_path = self.candidates_dir / "id_mapping.json"

        if candidate_index_path.exists() and candidate_map_path.exists():
            try:
                self.candidate_index = faiss.read_index(str(candidate_index_path))

                with open(candidate_map_path, 'r') as f:
                    mapping_data = json.load(f)
                    self.candidate_id_map = {int(k): v for k, v in mapping_data["id_map"].items()}
                    self.candidate_id_reverse_map = {v: int(k) for k, v in mapping_data["id_map"].items()}

                logger.info(f"Loaded candidate index with {self.candidate_index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Error loading candidate index: {str(e)}")
                # Initialize new index
                self.candidate_index = None

        # Load job index
        job_index_path = self.jobs_dir / "faiss_index.bin"
        job_map_path = self.jobs_dir / "id_mapping.json"

        if job_index_path.exists() and job_map_path.exists():
            try:
                self.job_index = faiss.read_index(str(job_index_path))

                with open(job_map_path, 'r') as f:
                    mapping_data = json.load(f)
                    self.job_id_map = {int(k): v for k, v in mapping_data["id_map"].items()}
                    self.job_id_reverse_map = {v: int(k) for k, v in mapping_data["id_map"].items()}

                logger.info(f"Loaded job index with {self.job_index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Error loading job index: {str(e)}")
                # Initialize new index
                self.job_index = None

    def _save_candidate_index(self) -> None:
        """Save the candidate index and ID mapping to disk."""
        if self.candidate_index is None:
            logger.warning("No candidate index to save")
            return

        try:
            # Save FAISS index
            index_path = self.candidates_dir / "faiss_index.bin"
            faiss.write_index(self.candidate_index, str(index_path))

            # Save ID mapping
            map_path = self.candidates_dir / "id_mapping.json"
            with open(map_path, 'w') as f:
                json.dump({
                    "id_map": self.candidate_id_map
                }, f)

            logger.info(f"Saved candidate index with {self.candidate_index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Error saving candidate index: {str(e)}")

    def _save_job_index(self) -> None:
        """Save the job index and ID mapping to disk."""
        if self.job_index is None:
            logger.warning("No job index to save")
            return

        try:
            # Save FAISS index
            index_path = self.jobs_dir / "faiss_index.bin"
            faiss.write_index(self.job_index, str(index_path))

            # Save ID mapping
            map_path = self.jobs_dir / "id_mapping.json"
            with open(map_path, 'w') as f:
                json.dump({
                    "id_map": self.job_id_map
                }, f)

            logger.info(f"Saved job index with {self.job_index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Error saving job index: {str(e)}")

    def _init_candidate_index(self, vector_dim: int) -> None:
        """Initialize the candidate index with the appropriate dimension."""
        # Use L2 distance for simplicity, can be changed to inner product (for cosine similarity)
        # if vectors are normalized
        self.candidate_index = faiss.IndexFlatL2(vector_dim)
        logger.info(f"Initialized new candidate index with dimension {vector_dim}")

    def _init_job_index(self, vector_dim: int) -> None:
        """Initialize the job index with the appropriate dimension."""
        # Use L2 distance for simplicity, can be changed to inner product (for cosine similarity)
        # if vectors are normalized
        self.job_index = faiss.IndexFlatL2(vector_dim)
        logger.info(f"Initialized new job index with dimension {vector_dim}")

    def store_candidate_embeddings(
            self,
            candidates_df: pd.DataFrame,
            id_column: str = 'Name ',
            embedding_column: str = 'hf_embedding'
    ) -> None:
        """
        Store candidate embeddings in the vector database.

        Args:
            candidates_df: DataFrame with candidate embeddings
            id_column: Column name for candidate identifier
            embedding_column: Column name for embeddings
        """
        # Skip if no embeddings to store
        if embedding_column not in candidates_df.columns or candidates_df.empty:
            logger.warning("No embeddings to store")
            return

        # Get vector dimension from first non-null embedding
        vectors_to_add = []
        ids_to_add = []

        for idx, row in candidates_df.iterrows():
            # Skip if embedding is None
            if row[embedding_column] is None:
                continue

            candidate_id = str(row[id_column]).strip()
            if not candidate_id:
                continue

            # Check if already in index
            if candidate_id in self.candidate_id_reverse_map:
                continue

            # Add to vectors and ids to be indexed
            vectors_to_add.append(np.array(row[embedding_column], dtype=np.float32))
            ids_to_add.append(candidate_id)

        # If nothing to add, return
        if not vectors_to_add:
            logger.info("No new candidate embeddings to add")
            return

        # Get vector dimension
        vector_dim = len(vectors_to_add[0])

        # Initialize index if needed
        if self.candidate_index is None:
            self._init_candidate_index(vector_dim)

        # Convert to numpy array
        embeddings_array = np.vstack(vectors_to_add).astype(np.float32)

        # Add vectors to index
        faiss.normalize_L2(embeddings_array)  # Normalize for cosine similarity

        # Get current size
        current_size = self.candidate_index.ntotal

        # Add vectors to index
        self.candidate_index.add(embeddings_array)

        # Update ID mapping
        for i, candidate_id in enumerate(ids_to_add):
            idx = current_size + i
            self.candidate_id_map[idx] = candidate_id
            self.candidate_id_reverse_map[candidate_id] = idx

        # Save to disk
        self._save_candidate_index()

        logger.info(f"Stored {len(vectors_to_add)} new candidate embeddings")

    def store_job_embeddings(
            self,
            jobs_df: pd.DataFrame,
            id_column: str = 'Role',
            embedding_column: str = 'hf_embedding'
    ) -> None:
        """
        Store job embeddings in the vector database.

        Args:
            jobs_df: DataFrame with job embeddings
            id_column: Column name for job identifier
            embedding_column: Column name for embeddings
        """
        # Skip if no embeddings to store
        if embedding_column not in jobs_df.columns or jobs_df.empty:
            logger.warning("No embeddings to store")
            return

        # Get vector dimension from first non-null embedding
        vectors_to_add = []
        ids_to_add = []

        for idx, row in jobs_df.iterrows():
            # Skip if embedding is None
            if row[embedding_column] is None:
                continue

            job_id = str(row[id_column]).strip()
            if not job_id:
                continue

            # Check if already in index
            if job_id in self.job_id_reverse_map:
                continue

            # Add to vectors and ids to be indexed
            vectors_to_add.append(np.array(row[embedding_column], dtype=np.float32))
            ids_to_add.append(job_id)

        # If nothing to add, return
        if not vectors_to_add:
            logger.info("No new job embeddings to add")
            return

        # Get vector dimension
        vector_dim = len(vectors_to_add[0])

        # Initialize index if needed
        if self.job_index is None:
            self._init_job_index(vector_dim)

        # Convert to numpy array
        embeddings_array = np.vstack(vectors_to_add).astype(np.float32)

        # Add vectors to index
        faiss.normalize_L2(embeddings_array)  # Normalize for cosine similarity

        # Get current size
        current_size = self.job_index.ntotal

        # Add vectors to index
        self.job_index.add(embeddings_array)

        # Update ID mapping
        for i, job_id in enumerate(ids_to_add):
            idx = current_size + i
            self.job_id_map[idx] = job_id
            self.job_id_reverse_map[job_id] = idx

        # Save to disk
        self._save_job_index()

        logger.info(f"Stored {len(vectors_to_add)} new job embeddings")

    def load_candidate_embeddings(
            self,
            candidates_df: pd.DataFrame,
            id_column: str = 'Name ',
            embedding_column: str = 'hf_embedding'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load stored embeddings for candidates.

        Args:
            candidates_df: DataFrame with candidate information
            id_column: Column name for candidate identifier
            embedding_column: Column name for embeddings

        Returns:
            DataFrame with embeddings added or updated and mask of rows needing embeddings
        """
        # Create a copy of the DataFrame
        result_df = candidates_df.copy()

        # Ensure embedding column exists
        if embedding_column not in result_df.columns:
            result_df[embedding_column] = None

        # Create mask for candidates that need new embeddings (initially all)
        needs_embedding = pd.Series(True, index=result_df.index)

        # If no index, all need embeddings
        if self.candidate_index is None:
            logger.info("No candidate index found, all candidates need embedding generation")
            return result_df, needs_embedding

        # Try to load stored embeddings
        for idx, row in result_df.iterrows():
            candidate_id = str(row[id_column]).strip()

            if not candidate_id:
                logger.warning(f"Empty candidate ID at index {idx}, skipping")
                continue

            # Check if in reverse map
            if candidate_id in self.candidate_id_reverse_map:
                # Get index position
                index_pos = self.candidate_id_reverse_map[candidate_id]

                # Get embedding
                embedding = self._get_candidate_embedding_by_index(index_pos)

                if embedding is not None:
                    result_df.at[idx, embedding_column] = embedding
                    needs_embedding.at[idx] = False

        loaded_count = (~needs_embedding).sum()
        logger.info(f"Loaded {loaded_count} stored candidate embeddings, {needs_embedding.sum()} need to be generated")

        return result_df, needs_embedding

    def load_job_embeddings(
            self,
            jobs_df: pd.DataFrame,
            id_column: str = 'Role',
            embedding_column: str = 'hf_embedding'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load stored embeddings for jobs.

        Args:
            jobs_df: DataFrame with job information
            id_column: Column name for job identifier
            embedding_column: Column name for embeddings

        Returns:
            DataFrame with embeddings added or updated and mask of rows needing embeddings
        """
        # Create a copy of the DataFrame
        result_df = jobs_df.copy()

        # Ensure embedding column exists
        if embedding_column not in result_df.columns:
            result_df[embedding_column] = None

        # Create mask for jobs that need new embeddings (initially all)
        needs_embedding = pd.Series(True, index=result_df.index)

        # If no index, all need embeddings
        if self.job_index is None:
            logger.info("No job index found, all jobs need embedding generation")
            return result_df, needs_embedding

        # Try to load stored embeddings
        for idx, row in result_df.iterrows():
            job_id = str(row[id_column]).strip()

            if not job_id:
                logger.warning(f"Empty job ID at index {idx}, skipping")
                continue

            # Check if in reverse map
            if job_id in self.job_id_reverse_map:
                # Get index position
                index_pos = self.job_id_reverse_map[job_id]

                # Get embedding
                embedding = self._get_job_embedding_by_index(index_pos)

                if embedding is not None:
                    result_df.at[idx, embedding_column] = embedding
                    needs_embedding.at[idx] = False

        loaded_count = (~needs_embedding).sum()
        logger.info(f"Loaded {loaded_count} stored job embeddings, {needs_embedding.sum()} need to be generated")

        return result_df, needs_embedding

    def _get_candidate_embedding_by_index(self, index: int) -> Optional[List[float]]:
        """
        Get a candidate embedding by index.

        Args:
            index: Index in the FAISS database

        Returns:
            Embedding vector or None if not found
        """
        try:
            # Check if index is valid
            if index >= self.candidate_index.ntotal:
                logger.warning(f"Invalid candidate index: {index}")
                return None

            # Reconstruct the vector
            embedding = np.zeros((1, self.candidate_index.d), dtype=np.float32)
            self.candidate_index.reconstruct(index, embedding[0])

            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"Error retrieving candidate embedding at index {index}: {str(e)}")
            return None

    def _get_job_embedding_by_index(self, index: int) -> Optional[List[float]]:
        """
        Get a job embedding by index.

        Args:
            index: Index in the FAISS database

        Returns:
            Embedding vector or None if not found
        """
        try:
            # Check if index is valid
            if index >= self.job_index.ntotal:
                logger.warning(f"Invalid job index: {index}")
                return None

            # Reconstruct the vector
            embedding = np.zeros((1, self.job_index.d), dtype=np.float32)
            self.job_index.reconstruct(index, embedding[0])

            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"Error retrieving job embedding at index {index}: {str(e)}")
            return None

    def search_similar_candidates(
            self,
            query_embedding: List[float],
            k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar candidates.

        Args:
            query_embedding: Embedding vector to search for
            k: Number of results to return

        Returns:
            List of dictionaries with candidate ID and similarity score
        """
        if self.candidate_index is None or self.candidate_index.ntotal == 0:
            logger.warning("No candidate index to search")
            return []

        try:
            # Convert query to numpy array
            query = np.array([query_embedding], dtype=np.float32)

            # Normalize query
            faiss.normalize_L2(query)

            # Search
            distances, indices = self.candidate_index.search(query, k)

            # Convert to results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1:  # FAISS uses -1 for "not found"
                    candidate_id = self.candidate_id_map.get(int(idx))
                    if candidate_id:
                        results.append({
                            "id": candidate_id,
                            "distance": float(distances[0][i]),
                            "similarity": 1.0 - float(distances[0][i]) / 2.0  # Convert L2 distance to similarity
                        })

            return results
        except Exception as e:
            logger.error(f"Error searching similar candidates: {str(e)}")
            return []

    def search_similar_jobs(
            self,
            query_embedding: List[float],
            k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar jobs.

        Args:
            query_embedding: Embedding vector to search for
            k: Number of results to return

        Returns:
            List of dictionaries with job ID and similarity score
        """
        if self.job_index is None or self.job_index.ntotal == 0:
            logger.warning("No job index to search")
            return []

        try:
            # Convert query to numpy array
            query = np.array([query_embedding], dtype=np.float32)

            # Normalize query
            faiss.normalize_L2(query)

            # Search
            distances, indices = self.job_index.search(query, k)

            # Convert to results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1:  # FAISS uses -1 for "not found"
                    job_id = self.job_id_map.get(int(idx))
                    if job_id:
                        results.append({
                            "id": job_id,
                            "distance": float(distances[0][i]),
                            "similarity": 1.0 - float(distances[0][i]) / 2.0  # Convert L2 distance to similarity
                        })

            return results
        except Exception as e:
            logger.error(f"Error searching similar jobs: {str(e)}")
            return []

    def calculate_similarity_matrix(
            self,
            candidates_df: pd.DataFrame,
            jobs_df: pd.DataFrame,
            candidate_id_column: str = 'Name ',
            job_id_column: str = 'Role'
    ) -> pd.DataFrame:
        """
        Calculate similarity matrix between candidates and jobs.

        Args:
            candidates_df: DataFrame with candidate information
            jobs_df: DataFrame with job information
            candidate_id_column: Column name for candidate identifier
            job_id_column: Column name for job identifier

        Returns:
            DataFrame with similarity scores
        """
        if self.candidate_index is None or self.job_index is None:
            logger.warning("Cannot calculate similarity matrix without indexes")
            return pd.DataFrame()

        try:
            # Get embeddings
            candidate_ids = candidates_df[candidate_id_column].tolist()
            job_ids = jobs_df[job_id_column].tolist()

            # Initialize similarity matrix
            similarity_matrix = np.zeros((len(candidate_ids), len(job_ids)))

            # Get vector dimensions
            candidate_dim = self.candidate_index.d
            job_dim = self.job_index.d

            # Check if dimensions match
            if candidate_dim != job_dim:
                logger.error(
                    f"Candidate embedding dimension ({candidate_dim}) does not match job embedding dimension ({job_dim})")
                return pd.DataFrame()

            # Get all vectors from indexes
            candidate_vectors = np.zeros((self.candidate_index.ntotal, candidate_dim), dtype=np.float32)
            for i in range(self.candidate_index.ntotal):
                self.candidate_index.reconstruct(i, candidate_vectors[i])

            job_vectors = np.zeros((self.job_index.ntotal, job_dim), dtype=np.float32)
            for i in range(self.job_index.ntotal):
                self.job_index.reconstruct(i, job_vectors[i])

            # For each candidate, calculate similarity with each job
            for i, candidate_id in enumerate(candidate_ids):
                if candidate_id not in self.candidate_id_reverse_map:
                    continue

                candidate_idx = self.candidate_id_reverse_map[candidate_id]
                candidate_vector = candidate_vectors[candidate_idx].reshape(1, -1)

                for j, job_id in enumerate(job_ids):
                    if job_id not in self.job_id_reverse_map:
                        continue

                    job_idx = self.job_id_reverse_map[job_id]
                    job_vector = job_vectors[job_idx].reshape(1, -1)

                    # Calculate cosine similarity
                    # Since vectors are normalized, cosine similarity = 1 - L2^2/2
                    dist = np.linalg.norm(candidate_vector - job_vector) ** 2
                    similarity = 1.0 - dist / 2.0

                    similarity_matrix[i, j] = similarity

            # Create DataFrame
            similarity_df = pd.DataFrame(
                similarity_matrix,
                index=candidate_ids,
                columns=job_ids
            )

            return similarity_df

        except Exception as e:
            logger.error(f"Error calculating similarity matrix: {str(e)}")
            return pd.DataFrame()