"""
Additional CLI commands for vector database management
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional


def add_vector_db_commands(subparsers):
    """
    Add vector database management commands to the CLI.

    Args:
        subparsers: Subparsers object from argparse
    """
    # 'vectordb' command group
    vectordb_parser = subparsers.add_parser(
        "vectordb",
        help="Vector database management commands"
    )

    # Create subcommands for vectordb
    vectordb_subparsers = vectordb_parser.add_subparsers(
        dest="vectordb_command",
        help="Vector database command to execute"
    )

    # 'info' command - Show information about the vector database
    info_parser = vectordb_subparsers.add_parser(
        "info",
        help="Show information about the vector database"
    )

    # 'clear' command - Clear the vector database
    clear_parser = vectordb_subparsers.add_parser(
        "clear",
        help="Clear the vector database"
    )
    clear_parser.add_argument(
        "--type",
        choices=["all", "candidates", "jobs"],
        default="all",
        help="Type of data to clear from vector database"
    )
    clear_parser.add_argument(
        "--force",
        action="store_true",
        help="Force clear without confirmation"
    )

    # 'rebuild' command - Rebuild the vector database from CSV files
    rebuild_parser = vectordb_subparsers.add_parser(
        "rebuild",
        help="Rebuild the vector database from CSV files"
    )
    rebuild_parser.add_argument(
        "--candidates",
        type=str,
        help="CSV file with candidate data",
        default=None
    )
    rebuild_parser.add_argument(
        "--jobs",
        type=str,
        help="CSV file with job listing data",
        default=None
    )


def handle_vectordb_command(args, app):
    """
    Handle vector database management commands.

    Args:
        args: Command-line arguments
        app: ResumeMatcherApp instance

    Returns:
        Exit code (0 for success, 1 for error)
    """
    if args.vectordb_command == "info":
        return command_vectordb_info(args, app)
    elif args.vectordb_command == "clear":
        return command_vectordb_clear(args, app)
    elif args.vectordb_command == "rebuild":
        return command_vectordb_rebuild(args, app)
    else:
        logging.error(f"Unknown vector database command: {args.vectordb_command}")
        return 1


def command_vectordb_info(args, app):
    """
    Show information about the vector database.

    Args:
        args: Command-line arguments
        app: ResumeMatcherApp instance

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Check if vector database is initialized
        if app.embedding_storage is None:
            logging.error("Vector database not initialized")
            return 1

        # Get database information
        candidate_count = (app.embedding_storage.candidate_index.ntotal
                           if app.embedding_storage.candidate_index is not None else 0)
        job_count = (app.embedding_storage.job_index.ntotal
                     if app.embedding_storage.job_index is not None else 0)

        # Print information
        print("\nVector Database Information:")
        print(f"  Storage directory: {app.embedding_storage.storage_dir}")
        print(f"  Candidates: {candidate_count}")
        print(f"  Jobs: {job_count}")

        # Print some example mappings if available
        if candidate_count > 0:
            print("\nSample Candidate Entries:")
            for i, (idx, name) in enumerate(app.embedding_storage.candidate_id_map.items()):
                if i >= 5:  # Show at most 5 examples
                    break
                print(f"  {idx}: {name}")

        if job_count > 0:
            print("\nSample Job Entries:")
            for i, (idx, name) in enumerate(app.embedding_storage.job_id_map.items()):
                if i >= 5:  # Show at most 5 examples
                    break
                print(f"  {idx}: {name}")

        return 0
    except Exception as e:
        logging.error(f"Error getting vector database info: {str(e)}", exc_info=True)
        return 1


def command_vectordb_clear(args, app):
    """
    Clear the vector database.

    Args:
        args: Command-line arguments
        app: ResumeMatcherApp instance

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Check if vector database is initialized
        if app.embedding_storage is None:
            logging.error("Vector database not initialized")
            return 1

        # Get type of data to clear
        clear_type = args.type

        # Check for confirmation unless --force is specified
        if not args.force:
            confirmation = input(f"Are you sure you want to clear {clear_type} data from the vector database? [y/N]: ")
            if confirmation.lower() != 'y':
                print("Operation cancelled")
                return 0

        # Clear the specified data
        if clear_type == "all" or clear_type == "candidates":
            # Initialize a new empty candidate index
            if app.embedding_storage.candidate_index is not None:
                dimension = app.embedding_storage.candidate_index.d
                app.embedding_storage.candidate_index = None
                app.embedding_storage._init_candidate_index(dimension)
                app.embedding_storage.candidate_id_map = {}
                app.embedding_storage.candidate_id_reverse_map = {}
                app.embedding_storage._save_candidate_index()
                print("Cleared candidate data from vector database")

        if clear_type == "all" or clear_type == "jobs":
            # Initialize a new empty job index
            if app.embedding_storage.job_index is not None:
                dimension = app.embedding_storage.job_index.d
                app.embedding_storage.job_index = None
                app.embedding_storage._init_job_index(dimension)
                app.embedding_storage.job_id_map = {}
                app.embedding_storage.job_id_reverse_map = {}
                app.embedding_storage._save_job_index()
                print("Cleared job data from vector database")

        return 0
    except Exception as e:
        logging.error(f"Error clearing vector database: {str(e)}", exc_info=True)
        return 1


def command_vectordb_rebuild(args, app):
    """
    Rebuild the vector database from CSV files.

    Args:
        args: Command-line arguments
        app: ResumeMatcherApp instance

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Check if vector database is initialized
        if app.embedding_storage is None:
            logging.error("Vector database not initialized")
            return 1

        # Override config paths if specified
        if args.candidates:
            app.config.file_paths.candidate_file = Path(args.candidates)

        if args.jobs:
            app.config.file_paths.listings_file = Path(args.jobs)

        # Load data
        candidates_df, jobs_df = app.load_data()

        # Prepare and embed data
        candidates_df, jobs_df = app.prepare_and_embed_data(candidates_df, jobs_df)

        print(f"Rebuilt vector database with {len(candidates_df)} candidates and {len(jobs_df)} jobs")

        return 0
    except Exception as e:
        logging.error(f"Error rebuilding vector database: {str(e)}", exc_info=True)
        return 1