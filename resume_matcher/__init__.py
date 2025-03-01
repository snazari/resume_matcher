"""
Resume Matcher - A system for matching candidate resumes to job listings using NLP.
"""

__version__ = "0.1.1"

# Import key classes for easier access
from resume_matcher.main import ResumeMatcherApp

# Expose the main application class as a direct import
__all__ = ["ResumeMatcherApp"]