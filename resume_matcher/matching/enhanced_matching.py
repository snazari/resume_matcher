"""
Enhanced matching engine that leverages vector database capabilities.
"""
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


def faiss_similarity_calculation(
        matching_engine,
        candidates_df: pd.DataFrame,
        jobs_df: pd.DataFrame,
        embedding_storage
) -> pd.DataFrame:
    """
    Calculate similarity matrix using FAISS vector database instead of in-memory computation.

    Args:
        matching_engine: The MatchingEngine instance
        candidates_df: DataFrame with candidate information (with embeddings)
        jobs_df: DataFrame with job information (with embeddings)
        embedding_storage: VectorDatabase instance

    Returns:
        DataFrame with similarity scores
    """
    logger.info("Using vector database for efficient similarity calculation")

    # Calculate similarity using vector database
    similarity_df = embedding_storage.calculate_similarity_matrix(
        candidates_df,
        jobs_df,
        candidate_id_column='Name ',
        job_id_column='Role'
    )

    # If the vector database calculation failed or returned empty dataframe,
    # fall back to the original method
    if similarity_df.empty:
        logger.warning("Vector database similarity calculation failed, falling back to original method")
        return matching_engine.calculate_similarity(candidates_df, jobs_df)

    return similarity_df


def find_top_jobs_for_candidate(
        candidate_embedding: List[float],
        embedding_storage,
        k: int = 5
) -> List[Dict]:
    """
    Find top job matches for a candidate embedding.

    Args:
        candidate_embedding: Embedding vector for the candidate
        embedding_storage: VectorDatabase instance
        k: Number of top matches to return

    Returns:
        List of dictionaries with job ID and similarity score
    """
    return embedding_storage.search_similar_jobs(candidate_embedding, k)


def find_top_candidates_for_job(
        job_embedding: List[float],
        embedding_storage,
        k: int = 5
) -> List[Dict]:
    """
    Find top candidate matches for a job embedding.

    Args:
        job_embedding: Embedding vector for the job
        embedding_storage: VectorDatabase instance
        k: Number of top matches to return

    Returns:
        List of dictionaries with candidate ID and similarity score
    """
    return embedding_storage.search_similar_candidates(job_embedding, k)