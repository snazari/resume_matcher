"""
Engine for matching candidates with job listings based on embeddings.
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

logger = logging.getLogger(__name__)

class MatchingEngine:
    """Engine for matching candidates with job listings based on embeddings."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize the matching engine.
        
        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = output_dir
    
    def prepare_candidate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare candidate data for matching.
        
        Args:
            df: DataFrame with candidate information
            
        Returns:
            DataFrame with combined text for embedding
        """
        logger.info("Preparing candidate data for matching")
        
        # Combine relevant columns
        columns_to_combine = ['Years of experience ', 'Degree', 'Resume Experience ']
        
        # Ensure all columns exist
        for col in columns_to_combine:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in candidate data")
                df[col] = ""  # Add empty column if missing
        
        # Combine text
        df['text_to_embed'] = df[columns_to_combine].astype(str).agg(' '.join, axis=1)
        
        return df
    
    def prepare_job_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare job listing data for matching.
        
        Args:
            df: DataFrame with job listing information
            
        Returns:
            DataFrame with combined text for embedding
        """
        logger.info("Preparing job listing data for matching")
        
        # Combine relevant columns
        columns_to_combine = ['Role', 'Description', 'Degree', 'Years of Exp', 'Expanded Experience', 'Notes']
        
        # Ensure all columns exist
        for col in columns_to_combine:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in job data")
                df[col] = ""  # Add empty column if missing
        
        # Combine text
        df['listing_to_embed'] = df[columns_to_combine].astype(str).agg(' '.join, axis=1)
        
        return df
    
    def calculate_similarity(
        self, 
        candidates_df: pd.DataFrame, 
        jobs_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate similarity between candidates and job listings.
        
        Args:
            candidates_df: DataFrame with candidate embeddings
            jobs_df: DataFrame with job listing embeddings
            
        Returns:
            DataFrame with similarity scores
        """
        logger.info("Calculating similarity between candidates and job listings")
        
        # Get embeddings
        candidate_embeddings = candidates_df['hf_embedding'].tolist()
        job_embeddings = jobs_df['hf_embedding'].tolist()
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(candidate_embeddings, job_embeddings)
        
        # Create similarity DataFrame
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=candidates_df['Name '],
            columns=jobs_df['Role']
        )
        
        return similarity_df
    
    def filter_candidates(
        self, 
        similarity_df: pd.DataFrame, 
        candidates_df: pd.DataFrame, 
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Filter candidates based on similarity threshold.
        
        Args:
            similarity_df: DataFrame with similarity scores
            candidates_df: DataFrame with candidate information
            threshold: Minimum similarity score to keep
            
        Returns:
            Filtered similarity DataFrame
        """
        logger.info(f"Filtering candidates with similarity threshold {threshold}")
        
        # Create a copy to avoid modifying the original
        filtered_df = similarity_df.copy()
        
        # Apply threshold
        filtered_df = filtered_df.where(filtered_df >= threshold)
        
        # Check for any additional filters in the candidates_df
        if 'Contingent' in candidates_df.columns:
            # Get contingent status for each candidate
            contingent_status = candidates_df.set_index('Name ')['Contingent']
            
            # Reindex to match similarity_df
            contingent_status = contingent_status.reindex(filtered_df.index)
            
            # Add as a new column to the similarity DataFrame
            filtered_df['Contingent'] = contingent_status
            
            logger.info("Added contingent status to similarity matrix")
        
        return filtered_df

    def visualize_similarity(self, similarity_df: pd.DataFrame) -> Tuple[Any, Any]:
        """
        Create visualizations for the similarity matrix.

        Args:
            similarity_df: DataFrame with similarity scores

        Returns:
            Tuple of (table figure, heatmap figure)
        """
        logger.info("Creating visualizations for similarity matrix")

        import plotly.graph_objects as go
        import plotly.express as px

        # Create table visualization
        table_fig = go.Figure(data=[go.Table(
            header=dict(
                values=[''] + list(similarity_df.columns),
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=[similarity_df.index] + [similarity_df[col] for col in similarity_df.columns],
                fill_color='lavender',
                align='left',
                format=[None] + ['.2f'] * len(similarity_df.columns)
            )
        )])

        table_fig.update_layout(
            title="Candidate-Job Similarity Matrix (Table)",
            width=1500,
            height=1000
        )

        # Create heatmap visualization
        heatmap_fig = px.imshow(
            similarity_df,
            labels=dict(x="Job Listing", y="Candidate", color="Similarity"),
            x=similarity_df.columns,
            y=similarity_df.index,
            color_continuous_scale="RdBu_r"
        )

        heatmap_fig.update_layout(
            title="Candidate-Job Similarity Matrix (Heatmap)",
            width=1500,
            height=1000
        )

        # Save the figures
        table_fig.write_html(str(self.output_dir / "similarity_table.html"))
        heatmap_fig.write_html(str(self.output_dir / "similarity_heatmap.html"))

        return table_fig, heatmap_fig

    def find_top_matches(self, similarity_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        """
        Find top matching candidates for each job.

        Args:
            similarity_df: DataFrame with similarity scores
            top_n: Number of top candidates to find per job

        Returns:
            DataFrame with top matches
        """
        logger.info(f"Finding top {top_n} matches per job")

        from resume_matcher.matching.candidate_ranker import CandidateRanker

        # Initialize candidate ranker
        ranker = CandidateRanker()

        # Rank candidates
        top_matches_df = ranker.rank_candidates(similarity_df, top_n=top_n)

        return top_matches_df

    def save_results(
            self,
            candidates_df: pd.DataFrame,
            jobs_df: pd.DataFrame,
            similarity_df: pd.DataFrame
    ) -> Dict[str, str]:
        """
        Save matching results to files.

        Args:
            candidates_df: DataFrame with candidate information
            jobs_df: DataFrame with job listing information
            similarity_df: DataFrame with similarity scores

        Returns:
            Dictionary with output file paths
        """
        logger.info("Saving matching results")

        # Create output paths
        similarity_csv_path = self.output_dir / "similarity_matrix.csv"
        candidates_csv_path = self.output_dir / "candidates_with_embeddings.csv"
        jobs_csv_path = self.output_dir / "jobs_with_embeddings.csv"

        # Save files
        similarity_df.to_csv(similarity_csv_path)

        # Save candidates and jobs without the embedding column to avoid large files
        candidates_to_save = candidates_df.drop(columns=['hf_embedding'], errors='ignore')
        jobs_to_save = jobs_df.drop(columns=['hf_embedding'], errors='ignore')

        candidates_to_save.to_csv(candidates_csv_path, index=False)
        jobs_to_save.to_csv(jobs_csv_path, index=False)

        # Return file paths
        return {
            "similarity_matrix": str(similarity_csv_path),
            "candidates": str(candidates_csv_path),
            "jobs": str(jobs_csv_path)
        }