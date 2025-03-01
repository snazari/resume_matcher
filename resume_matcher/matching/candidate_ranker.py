"""
Module for ranking and prioritizing candidates based on similarity scores.
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class CandidateRanker:
    """Class for ranking and prioritizing candidates based on similarity scores."""
    
    def __init__(self):
        """Initialize the candidate ranker."""
        pass
    
    def rank_candidates(
        self, 
        similarity_df: pd.DataFrame, 
        top_n: int = 5
    ) -> pd.DataFrame:
        """
        Rank candidates for each job based on similarity scores.
        
        Args:
            similarity_df: DataFrame with similarity scores
            top_n: Number of top candidates to consider for each job
            
        Returns:
            DataFrame with ranked candidates for each job
        """
        logger.info(f"Ranking top {top_n} candidates for each job")
        
        # Transpose to get jobs as rows
        df_transposed = similarity_df.T
        
        results = []
        
        # For each job, find top candidates
        for job in df_transposed.index:
            # Get scores for this job
            job_scores = df_transposed.loc[job]
            
            # Sort and get top candidates
            top_candidates = job_scores.sort_values(ascending=False).head(top_n)
            
            # Add to results
            for rank, (candidate, score) in enumerate(top_candidates.items(), 1):
                results.append({
                    "Job": job,
                    "Candidate": candidate,
                    "Similarity": score,
                    "Rank": rank
                })
        
        # Create DataFrame from results
        ranked_df = pd.DataFrame(results)
        
        return ranked_df
    
    def optimize_assignments(
        self, 
        similarity_df: pd.DataFrame,
        max_jobs_per_candidate: int = 2
    ) -> pd.DataFrame:
        """
        Optimize job assignments to prevent one candidate from getting too many offers.
        
        Args:
            similarity_df: DataFrame with similarity scores
            max_jobs_per_candidate: Maximum number of jobs to assign to each candidate
            
        Returns:
            DataFrame with optimized job assignments
        """
        logger.info("Optimizing job assignments")
        
        # Create a copy of the similarity matrix
        scores = similarity_df.copy()
        
        # Initialize results
        assignments = []
        candidates_assigned = {}  # Keep track of how many jobs each candidate has
        
        # Continue until all jobs have a candidate or no candidates are available
        while scores.size > 0 and scores.max().max() > 0:
            # Find the highest similarity score
            job_idx = scores.max().idxmax()  # Job with the highest similarity
            candidate_idx = scores[job_idx].idxmax()  # Best candidate for that job
            score = scores.at[candidate_idx, job_idx]
            
            # Record the assignment
            assignments.append({
                "Job": job_idx,
                "Candidate": candidate_idx,
                "Similarity": score
            })
            
            # Update candidate assignment count
            candidates_assigned[candidate_idx] = candidates_assigned.get(candidate_idx, 0) + 1
            
            # Remove the job from consideration
            scores.drop(job_idx, axis=1, inplace=True)
            
            # If candidate has reached maximum jobs, remove them from consideration
            if candidates_assigned.get(candidate_idx, 0) >= max_jobs_per_candidate:
                scores.drop(candidate_idx, axis=0, inplace=True)
        
        # Create DataFrame from assignments
        assignments_df = pd.DataFrame(assignments)
        
        return assignments_df
    
    def analyze_rankings(self, ranked_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze candidate rankings for insights.
        
        Args:
            ranked_df: DataFrame with ranked candidates
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Analyzing candidate rankings")
        
        # Number of unique candidates
        unique_candidates = ranked_df['Candidate'].nunique()
        
        # Number of unique jobs
        unique_jobs = ranked_df['Job'].nunique()
        
        # Average similarity score
        avg_similarity = ranked_df['Similarity'].mean()
        
        # Most versatile candidates (appear for multiple jobs)
        candidate_counts = ranked_df['Candidate'].value_counts()
        versatile_candidates = candidate_counts[candidate_counts > 1].to_dict()
        
        # Most competitive jobs (highest average similarity)
        job_avg_similarity = ranked_df.groupby('Job')['Similarity'].mean()
        competitive_jobs = job_avg_similarity.sort_values(ascending=False).to_dict()
        
        # Return analysis results
        return {
            "unique_candidates": unique_candidates,
            "unique_jobs": unique_jobs,
            "avg_similarity": avg_similarity,
            "versatile_candidates": versatile_candidates,
            "competitive_jobs": competitive_jobs
        }
