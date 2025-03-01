"""
Module for loading and parsing different document types.
"""
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Class for loading and processing various document types."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document loader.
        
        Args:
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )
    
    def load_document(self, filepath: Union[str, Path]) -> Optional[str]:
        """
        Load and extract text from a document file.
        
        Args:
            filepath: Path to the document file
            
        Returns:
            Extracted text content or None if loading failed
        """
        filepath = Path(filepath)
        
        try:
            # Determine file type and use appropriate loader
            file_extension = filepath.suffix.lower()
            
            if file_extension == ".pdf":
                return self._load_pdf(filepath)
            elif file_extension == ".docx":
                return self._load_docx(filepath)
            elif file_extension in [".csv", ".xlsx", ".xls"]:
                return self._load_tabular(filepath)
            else:
                logger.warning(f"Unsupported file type: {file_extension}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading document {filepath.name}: {str(e)}")
            return None
    
    def _load_pdf(self, filepath: Path) -> Optional[str]:
        """
        Load text from a PDF file.
        
        Args:
            filepath: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            loader = PyPDFLoader(str(filepath))
            docs = loader.load()
            
            # Combine all pages
            text = ""
            for doc in docs:
                text += doc.page_content + "\n\n"
                
            return text
        
        except Exception as e:
            logger.error(f"Error loading PDF {filepath.name}: {str(e)}")
            return None
    
    def _load_docx(self, filepath: Path) -> Optional[str]:
        """
        Load text from a DOCX file.
        
        Args:
            filepath: Path to the DOCX file
            
        Returns:
            Extracted text content
        """
        try:
            loader = Docx2txtLoader(str(filepath))
            docs = loader.load()
            
            # Combine all content
            text = ""
            for doc in docs:
                text += doc.page_content + "\n\n"
                
            return text
        
        except Exception as e:
            logger.error(f"Error loading DOCX {filepath.name}: {str(e)}")
            return None
    
    def _load_tabular(self, filepath: Path) -> Optional[str]:
        """
        Load text from a tabular file (CSV, Excel).
        
        Args:
            filepath: Path to the tabular file
            
        Returns:
            Text representation of the tabular data
        """
        try:
            file_extension = filepath.suffix.lower()
            
            if file_extension == ".csv":
                df = pd.read_csv(filepath)
            elif file_extension in [".xlsx", ".xls"]:
                df = pd.read_excel(filepath)
            else:
                logger.warning(f"Unsupported tabular file type: {file_extension}")
                return None
            
            # Convert DataFrame to string representation
            return df.to_string()
        
        except Exception as e:
            logger.error(f"Error loading tabular file {filepath.name}: {str(e)}")
            return None
    
    def load_and_split(self, filepath: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load a document and split it into chunks.
        
        Args:
            filepath: Path to the document file
            
        Returns:
            List of text chunks with metadata
        """
        filepath = Path(filepath)
        
        # Load the document
        text = self.load_document(filepath)
        if not text:
            return []
        
        # Split text into chunks
        try:
            chunks = self.text_splitter.create_documents([text])
            
            # Convert to dictionaries
            result = []
            for i, chunk in enumerate(chunks):
                result.append({
                    "text": chunk.page_content,
                    "metadata": {
                        "source": filepath.name,
                        "chunk_id": i,
                        "start_index": chunk.metadata.get("start_index", 0)
                    }
                })
            
            return result
        
        except Exception as e:
            logger.error(f"Error splitting document {filepath.name}: {str(e)}")
            return []
    
    def load_dataframe(self, filepath: Union[str, Path]) -> Optional[pd.DataFrame]:
        """
        Load tabular data into a pandas DataFrame.
        
        Args:
            filepath: Path to the tabular file
            
        Returns:
            DataFrame or None if loading failed
        """
        filepath = Path(filepath)
        
        try:
            file_extension = filepath.suffix.lower()
            
            if file_extension == ".csv":
                return pd.read_csv(filepath)
            elif file_extension in [".xlsx", ".xls"]:
                return pd.read_excel(filepath)
            else:
                logger.warning(f"Unsupported tabular file type: {file_extension}")
                return None
        
        except Exception as e:
            logger.error(f"Error loading dataframe from {filepath.name}: {str(e)}")
            return None
