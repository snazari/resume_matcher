#!/usr/bin/env python3
"""
Command-line interface for the Resume Matcher application.
Provides tools for processing resumes, matching candidates to job listings,
and visualizing results.
"""
import argparse
import sys
import logging
import os
from pathlib import Path
import json
from typing import List, Optional, Dict, Any

from resume_matcher import ResumeMatcherApp, __version__
from resume_matcher.utils.file_utils import ensure_directory_exists


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('resume_matcher.log')
        ]
    )


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Resume Matcher - Match candidate resumes to job listings using NLP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"Resume Matcher v{__version__}"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Path to configuration file",
        default=os.environ.get("RESUME_MATCHER_CONFIG", "config.yaml")
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # 'process' command - Process resumes to extract information
    process_parser = subparsers.add_parser(
        "process",
        help="Process resume files to extract information"
    )
    process_parser.add_argument(
        "resume_dir",
        type=str,
        help="Directory containing resume files (PDF and DOCX)"
    )
    process_parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output CSV file for extracted data",
        default="candidates.csv"
    )

    # 'match' command - Match candidates to job listings
    match_parser = subparsers.add_parser(
        "match",
        help="Match candidates to job listings"
    )
    match_parser.add_argument(
        "--candidates",
        type=str,
        help="CSV file with candidate data",
        default=None
    )
    match_parser.add_argument(
        "--jobs",
        type=str,
        help="CSV file with job listing data",
        default=None
    )
    match_parser.add_argument(
        "-o", "--output-dir",
        type=str,
        help="Directory for output files",
        default="output"
    )
    match_parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualization generation"
    )

    # 'pipeline' command - Run the full pipeline
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run the full resume matching pipeline"
    )
    pipeline_parser.add_argument(
        "--process-resumes",
        action="store_true",
        help="Process resume files instead of using existing CSV"
    )
    pipeline_parser.add_argument(
        "--resume-dir",
        type=str,
        help="Directory containing resume files (required if --process-resumes is used)"
    )
    pipeline_parser.add_argument(
        "-o", "--output-dir",
        type=str,
        help="Directory for output files",
        default="output"
    )
    pipeline_parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualization generation"
    )

    # Add vector database commands
    from resume_matcher.cli.vector_db_commands import add_vector_db_commands
    add_vector_db_commands(subparsers)

    return parser.parse_args(args)


def command_process(args: argparse.Namespace) -> int:
    """Handle the 'process' command."""
    try:
        # Initialize the application with config_path
        app = ResumeMatcherApp(config_path=args.config)

        # Process resumes
        resume_dir = Path(args.resume_dir)
        if not resume_dir.exists() or not resume_dir.is_dir():
            logging.error(f"Resume directory not found: {resume_dir}")
            return 1

        logging.info(f"Processing resumes from {resume_dir}")
        candidates_df = app.process_resumes(str(resume_dir))

        # Save to output file
        output_path = Path(args.output)
        ensure_directory_exists(output_path.parent)
        candidates_df.to_csv(output_path, index=False)
        logging.info(f"Saved extracted candidate data to {output_path}")

        return 0
    except Exception as e:
        logging.error(f"Error processing resumes: {str(e)}", exc_info=True)
        return 1


def command_match(args: argparse.Namespace) -> int:
    """Handle the 'match' command."""
    try:
        # Initialize the application with config_path
        app = ResumeMatcherApp(config_path=args.config)

        # Ensure output directory exists
        output_dir = Path(args.output_dir)
        ensure_directory_exists(output_dir)

        # Override config paths if specified
        if args.candidates:
            app.config.file_paths.candidate_file = Path(args.candidates)

        if args.jobs:
            app.config.file_paths.listings_file = Path(args.jobs)

        # Load data
        candidates_df, jobs_df = app.load_data()

        # Prepare and embed data
        candidates_df, jobs_df = app.prepare_and_embed_data(candidates_df, jobs_df)

        # Match candidates to jobs
        results = app.match_candidates_to_jobs(
            candidates_df,
            jobs_df,
            visualize=not args.no_viz
        )

        # Print summary
        print("\nTop 5 matches per job:")
        for job, candidates in results["top_matches"].groupby("Job"):
            print(f"\nJob: {job}")
            for i, (_, row) in enumerate(candidates.head(5).iterrows(), 1):
                print(f"  {i}. {row['Candidate']} - Score: {row['Similarity']:.2f}")

        print(f"\nResults saved to {output_dir}")
        for file_type, file_path in results["output_paths"].items():
            print(f"  - {file_type}: {file_path}")

        return 0
    except Exception as e:
        logging.error(f"Error matching candidates: {str(e)}", exc_info=True)
        return 1


def command_pipeline(args: argparse.Namespace) -> int:
    """Handle the 'pipeline' command."""
    try:
        # Initialize the application with config_path
        app = ResumeMatcherApp(config_path=args.config)

        # Ensure output directory exists
        output_dir = Path(args.output_dir)
        ensure_directory_exists(output_dir)
        app.config.file_paths.output_dir = output_dir

        # Check if we need to process resumes
        if args.process_resumes:
            if not args.resume_dir:
                logging.error("Resume directory must be specified when using --process-resumes")
                return 1

            resume_dir = Path(args.resume_dir)
            if not resume_dir.exists() or not resume_dir.is_dir():
                logging.error(f"Resume directory not found: {resume_dir}")
                return 1

        # Run the full pipeline
        results = app.run_full_pipeline(
            process_resumes=args.process_resumes,
            resume_dir=args.resume_dir,
            visualize=not args.no_viz
        )

        # Print summary
        print("\nTop 5 matches per job:")
        for job, candidates in results["top_matches"].groupby("Job"):
            print(f"\nJob: {job}")
            for i, (_, row) in enumerate(candidates.head(5).iterrows(), 1):
                print(f"  {i}. {row['Candidate']} - Score: {row['Similarity']:.2f}")

        print(f"\nResults saved to {output_dir}")
        for file_type, file_path in results["output_paths"].items():
            print(f"  - {file_type}: {file_path}")

        return 0
    except Exception as e:
        logging.error(f"Error in pipeline: {str(e)}", exc_info=True)
        return 1


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI."""
    parsed_args = parse_args(args)

    # Set up logging
    setup_logging(parsed_args.verbose)

    # Handle commands
    if parsed_args.command == "process":
        return command_process(parsed_args)
    elif parsed_args.command == "match":
        return command_match(parsed_args)
    elif parsed_args.command == "pipeline":
        return command_pipeline(parsed_args)
    elif parsed_args.command == "vectordb":
        # Initialize the application with config_path
        app = ResumeMatcherApp(config_path=parsed_args.config)
        from resume_matcher.cli.vector_db_commands import handle_vectordb_command
        return handle_vectordb_command(parsed_args, app)
    else:
        logging.error("No command specified. Use --help for usage information.")
        return 1


if __name__ == "__main__":
    sys.exit(main())