"""
Utility functions for file operations in the Resume Matcher system.
"""
import os
import glob
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def get_file_list(
    directory_path: str, 
    extensions: List[str] = ['.pdf', '.docx']
) -> List[Tuple[str, str]]:
    """
    Get a list of files with specified extensions from a directory.
    
    Args:
        directory_path: Path to the directory
        extensions: List of file extensions to include
        
    Returns:
        List of (filename, filepath) tuples
    """
    files = []
    
    try:
        # Ensure the directory exists
        if not os.path.exists(directory_path):
            logger.error(f"Directory does not exist: {directory_path}")
            return []
        
        # Get all files with the specified extensions
        for ext in extensions:
            pattern = os.path.join(directory_path, f"*{ext}")
            for filepath in glob.glob(pattern):
                filename = os.path.basename(filepath)
                files.append((filename, filepath))
        
        logger.info(f"Found {len(files)} files with extensions {extensions} in {directory_path}")
        return files
    
    except Exception as e:
        logger.error(f"Error getting file list from {directory_path}: {str(e)}")
        return []


def ensure_directory_exists(directory_path: Path) -> bool:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        True if the directory exists or was created, False otherwise
    """
    try:
        directory_path = Path(directory_path)
        if not directory_path.exists():
            directory_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating directory {directory_path}: {str(e)}")
        return False


def get_latest_file(
    directory_path: str, 
    pattern: str = "*", 
    return_path: bool = True
) -> Optional[str]:
    """
    Get the most recently modified file matching a pattern in a directory.
    
    Args:
        directory_path: Path to the directory
        pattern: Glob pattern to match files
        return_path: Whether to return the full path or just the filename
        
    Returns:
        Path or filename of the most recent file, or None if no files found
    """
    try:
        # Ensure the directory exists
        if not os.path.exists(directory_path):
            logger.error(f"Directory does not exist: {directory_path}")
            return None
        
        # Get all files matching the pattern
        pattern_path = os.path.join(directory_path, pattern)
        files = glob.glob(pattern_path)
        
        if not files:
            logger.warning(f"No files matching pattern '{pattern}' found in {directory_path}")
            return None
        
        # Find the most recent file
        latest_file = max(files, key=os.path.getmtime)
        
        if return_path:
            return latest_file
        else:
            return os.path.basename(latest_file)
    
    except Exception as e:
        logger.error(f"Error getting latest file from {directory_path}: {str(e)}")
        return None
