"""
Data handling components for the Resume Matcher system.
"""

from resume_matcher.data.embedding_service import EmbeddingService
from resume_matcher.data.document_loader import DocumentLoader
from resume_matcher.data.data_processor import DataProcessor

__all__ = ["EmbeddingService", "DocumentLoader", "DataProcessor"]