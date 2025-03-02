import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dotenv import load_dotenv
from resume_matcher.config.config_manager import ConfigManager, AppConfig
from resume_matcher.data.embedding_service import EmbeddingService
from resume_matcher.data.embedding_storage import EmbeddingStorage
from resume_matcher.extraction.resume_extractor import ResumeExtractor
from resume_matcher.matching.matching_engine import MatchingEngine


class ResumeMatcherApp:
    """Main application class for Resume Matcher system."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Resume Matcher application.

        Args:
            config_path: Path to configuration file
        """
        # Set up logging
        self._setup_logging()

        # Load configuration
        self.logger.info("Sam Nazari, Ph.D.")
        self.logger.info("Initializing Resume Matcher application")
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()

        # Validate configuration
        self.config_manager.validate_paths()

        # Initialize components
        self._init_components()

    def _setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('resume_matcher.log')
            ]
        )
        self.logger = logging.getLogger(__name__)

    """
    Updated methods for ResumeMatcherApp to correctly use embedding storage
    """

    def _init_components(self):
        """Initialize application components."""
        # Set up logging
        logging.basicConfig(level=logging.INFO)

        # Create embedding storage
        storage_dir = Path(self.config.file_paths.output_dir) / "embeddings"
        from resume_matcher.data.simplified_embedding_storage import SimpleEmbeddingStorage
        self.embedding_storage = SimpleEmbeddingStorage(storage_dir)

        # Create embedding service
        self.embedding_service = EmbeddingService(
            api_url=self.config.huggingface.api_url,
            api_token=self.config.huggingface.api_token
        )

        # Create matching engine
        self.matching_engine = MatchingEngine(
            output_dir=self.config.file_paths.output_dir
        )

        # Create resume extractor if needed
        if hasattr(self.config, 'llm') and self.config.llm:
            self.resume_extractor = ResumeExtractor(
                llm_model_id=self.config.llm.model_id,
                api_token=self.config.llm.api_token
            )
        else:
            self.resume_extractor = None

            # Test embedding storage functionality
        if self.config.debug_mode:
            self.logger.info("Debug mode is enabled, testing embedding storage")
            storage_working = self.test_embedding_storage()
            if storage_working:
                self.logger.info("CHECK MARK: Embedding storage test passed - storage is working correctly")
            else:
                self.logger.warning("NO CHECK MARK: Embedding storage test failed - storage may not be working")

    def prepare_and_embed_data(
            self,
            candidates_df: pd.DataFrame,
            jobs_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.logger.info("Preparing and embedding data")
        candidates_df = self.matching_engine.prepare_candidate_data(candidates_df)
        jobs_df = self.matching_engine.prepare_job_data(jobs_df)
        print(candidates_df)
        print(jobs_df)
        self.logger.info("Checking for stored embeddings...")

        # Handle Candidate Embeddings
        self.logger.info("Loading candidate embeddings from storage...")
        candidates_df, candidates_mask = self.embedding_storage.load_embeddings(
            candidates_df,
            id_column='Name ',
            embedding_column='hf_embedding',
            is_candidate=True
        )
        self.logger.info(f"Need to generate embeddings for {candidates_mask.sum()} of {len(candidates_df)} candidates")
        if candidates_mask.any():
            self.logger.info(f"Generating embeddings for {candidates_mask.sum()} candidates")
            texts_to_embed = candidates_df.loc[candidates_mask, 'text_to_embed'].tolist()
            new_embeddings = self.embedding_service.generate_embeddings(texts_to_embed)
            if new_embeddings:
                idx_list = candidates_df.loc[candidates_mask].index.tolist()
                for i, idx in enumerate(idx_list):
                    candidates_df.at[idx, 'hf_embedding'] = new_embeddings[i]
                self.logger.info("Storing new candidate embeddings...")
                self.embedding_storage.store_embeddings(
                    candidates_df.loc[candidates_mask],
                    id_column='Name ',
                    embedding_column='hf_embedding',
                    is_candidate=True
                )
            else:
                self.logger.warning("Failed to generate embeddings for candidates")
        else:
            self.logger.info("All candidate embeddings loaded from storage - no new embeddings needed")

        # Handle Job Embeddings
        self.logger.info("Loading job embeddings from storage...")
        jobs_df, jobs_mask = self.embedding_storage.load_embeddings(
            jobs_df,
            id_column='Role',
            embedding_column='hf_embedding',
            is_candidate=False
        )
        self.logger.info(f"Need to generate embeddings for {jobs_mask.sum()} of {len(jobs_df)} jobs")
        if jobs_mask.any():
            self.logger.info(f"Generating embeddings for {jobs_mask.sum()} job listings")
            texts_to_embed = jobs_df.loc[jobs_mask, 'listing_to_embed'].tolist()
            new_embeddings = self.embedding_service.generate_embeddings(texts_to_embed)
            if new_embeddings:
                idx_list = jobs_df.loc[jobs_mask].index.tolist()
                for i, idx in enumerate(idx_list):
                    jobs_df.at[idx, 'hf_embedding'] = new_embeddings[i]
                self.logger.info("Storing new job embeddings...")
                self.embedding_storage.store_embeddings(
                    jobs_df.loc[jobs_mask],
                    id_column='Role',
                    embedding_column='hf_embedding',
                    is_candidate=False
                )
            else:
                self.logger.warning("Failed to generate embeddings for jobs")
        else:
            self.logger.info("All job embeddings loaded from storage - no new embeddings needed")

        return candidates_df, jobs_df
    def process_resumes(self, directory_path: str) -> pd.DataFrame:
        """
        Process resume files to extract information.

        Args:
            directory_path: Path to directory containing resume files

        Returns:
            DataFrame with extracted resume information
        """
        self.logger.info(f"Processing resumes from {directory_path}")

        if self.resume_extractor is None:
            raise ValueError("Resume extractor not configured")

        candidates_df = self.resume_extractor.process_directory(directory_path)

        if candidates_df.empty:
            self.logger.warning("No candidate data extracted")
            return pd.DataFrame()

        # Save results
        output_path = self.config.file_paths.output_dir / "extracted_candidates.csv"
        candidates_df.to_csv(output_path, index=False)
        self.logger.info(f"Saved extracted candidate data to {output_path}")

        return candidates_df

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load candidate and job listing data from files.

        Returns:
            Tuple of (candidates DataFrame, jobs DataFrame)
        """
        self.logger.info("Loading data from files")

        # Load candidate data
        candidates_df = pd.read_csv(self.config.file_paths.candidate_file)
        self.logger.info(f"Loaded {len(candidates_df)} candidates")

        # Load job listing data
        jobs_df = pd.read_csv(self.config.file_paths.listings_file)
        self.logger.info(f"Loaded {len(jobs_df)} job listings")

        return candidates_df, jobs_df

    def prepare_and_embed_data(
            self,
            candidates_df: pd.DataFrame,
            jobs_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare and generate embeddings for candidate and job data.

        Args:
            candidates_df: DataFrame with candidate information
            jobs_df: DataFrame with job listing information

        Returns:
            Tuple of (candidates DataFrame with embeddings, jobs DataFrame with embeddings)
        """
        self.logger.info("Preparing and embedding data")

        # Prepare candidate data
        candidates_df = self.matching_engine.prepare_candidate_data(candidates_df)

        # Prepare job data
        jobs_df = self.matching_engine.prepare_job_data(jobs_df)

        # Generate embeddings for candidates
        self.logger.info("Generating embeddings for candidates")
        candidates_df = self.embedding_service.embed_dataframe(
            candidates_df,
            text_column='text_to_embed',
            embedding_column='hf_embedding'
        )

        # Generate embeddings for jobs
        self.logger.info("Generating embeddings for job listings")
        jobs_df = self.embedding_service.embed_dataframe(
            jobs_df,
            text_column='listing_to_embed',
            embedding_column='hf_embedding'
        )

        return candidates_df, jobs_df

    def match_candidates_to_jobs(
            self,
            candidates_df: pd.DataFrame,
            jobs_df: pd.DataFrame,
            visualize: bool = True
    ) -> Dict[str, Any]:
        """
        Match candidates to job listings.

        Args:
            candidates_df: DataFrame with candidate embeddings
            jobs_df: DataFrame with job listing embeddings
            visualize: Whether to create visualizations

        Returns:
            Dictionary with results and output file paths
        """
        self.logger.info("Matching candidates to jobs")

        # Calculate similarity
        similarity_df = self.matching_engine.calculate_similarity(
            candidates_df,
            jobs_df
        )

        # Create visualizations if requested
        if visualize:
            table_fig, heatmap_fig = self.matching_engine.visualize_similarity(similarity_df)

            # Display figures
            table_fig.show()
            heatmap_fig.show()

        # Find top matches
        top_matches_df = self.matching_engine.find_top_matches(similarity_df)

        # Save results
        output_paths = self.matching_engine.save_results(
            candidates_df,
            jobs_df,
            similarity_df
        )

        # Add top matches to output paths
        top_matches_path = self.config.file_paths.output_dir / "top_matches.csv"
        top_matches_df.to_csv(top_matches_path, index=False)
        output_paths["top_matches"] = str(top_matches_path)

        return {
            "similarity_matrix": similarity_df,
            "top_matches": top_matches_df,
            "output_paths": output_paths
        }

    def run_full_pipeline(
            self,
            process_resumes: bool = False,
            resume_dir: Optional[str] = None,
            visualize: bool = True
    ) -> Dict[str, Any]:
        """
        Run the full resume matching pipeline.

        Args:
            process_resumes: Whether to process resume files
            resume_dir: Directory containing resume files (required if process_resumes is True)
            visualize: Whether to create visualizations

        Returns:
            Dictionary with results and output file paths
        """
        self.logger.info("Running full resume matching pipeline")

        # Process resumes if requested
        if process_resumes:
            if resume_dir is None:
                raise ValueError("Resume directory must be provided when process_resumes is True")

            candidates_df = self.process_resumes(resume_dir)

            # Need to also load jobs_df since it wasn't set above
            _, jobs_df = self.load_data()
        else:
            # Load data from files
            candidates_df, jobs_df = self.load_data()

        # Prepare and embed data
        candidates_df, jobs_df = self.prepare_and_embed_data(
            candidates_df,
            jobs_df
        )

        # Match candidates to jobs
        results = self.match_candidates_to_jobs(
            candidates_df,
            jobs_df,
            visualize=visualize
        )

        self.logger.info("Resume matching pipeline completed successfully")

        return results

    def test_embedding_storage(self):
        """Test that embedding storage is working properly."""
        self.logger.info("Testing embedding storage...")

        # Load a test file
        test_df = pd.DataFrame({
            'id': ['test1', 'test2'],
            'text': ['This is a test', 'Another test'],
            'embedding': [None, None]
        })

        # Generate embeddings
        texts = test_df['text'].tolist()
        embeddings = self.embedding_service.generate_embeddings(texts)

        # Store embeddings
        for i, idx in enumerate(test_df.index):
            test_df.at[idx, 'embedding'] = embeddings[i]

        # Store to disk
        self.embedding_storage.store_embeddings(
            test_df,
            id_column='id',
            embedding_column='embedding',
            is_candidate=True
        )

        # Now try to load them
        loaded_df, mask = self.embedding_storage.load_embeddings(
            test_df.copy(),
            id_column='id',
            embedding_column='embedding',
            is_candidate=True
        )

        # Check results
        self.logger.info(f"Test complete. Mask shows {mask.sum()} of {len(mask)} need generating")
        return mask.sum() == 0  # Should be 0 if all loaded correctly

if __name__ == "__main__":
    import os
    from pathlib import Path

    # Get the directory where main.py is located
    base_dir = Path(__file__).parent.parent.absolute()

    # Load the .env file explicitly from that location
    dotenv_path = base_dir / '.env'
    print(f"Looking for .env at: {dotenv_path} (exists: {dotenv_path.exists()})")

    load_dotenv(dotenv_path=dotenv_path)
    print("HUGGINGFACE_TOKEN in environment:", "HUGGINGFACE_TOKEN" in os.environ)
    if "HUGGINGFACE_TOKEN" in os.environ:
        print("Token starts with:", os.environ["HUGGINGFACE_TOKEN"][:5] + "...")

    # Example usage
    app = ResumeMatcherApp()
    results = app.run_full_pipeline()

    print("Top matches:")
    print(results["top_matches"].head(10))
# """
# Main application module for Resume Matcher.
# """
# import logging
#
#
# class ResumeMatcherApp:
#     """Placeholder for the main application class."""
#
#     def __init__(self, config_path=None):
#         """Initialize the application with minimal functionality."""
#         self.logger = logging.getLogger(__name__)
#         self.logger.info("Initializing Resume Matcher (minimal version)")
#
#     def run_full_pipeline(self, **kwargs):
#         """Placeholder for the full pipeline."""
#         self.logger.info("Full pipeline not implemented yet")
#         return {"status": "not_implemented"}