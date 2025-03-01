"""
Module for extracting structured information from resume files.
"""
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import InferenceClient
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CandidateInfo(BaseModel):
    """Data model for candidate information extracted from a resume."""
    name: str = Field(..., description="Full name of the candidate")
    years_of_experience: str = Field(..., description="Years of professional experience")
    degree: str = Field(..., description="Academic degrees earned by the candidate")
    experience_summary: str = Field(..., description="Summary of candidate's professional experience")
    source_file: str = Field(..., description="Source resume filename")


class ResumeExtractor:
    """Extracts structured information from resume files using LLMs."""
    
    def __init__(self, llm_model_id: str, api_token: str, max_workers: int = 4):
        """
        Initialize the resume extractor.
        
        Args:
            llm_model_id: Hugging Face model ID for text generation
            api_token: API token for Hugging Face
            max_workers: Maximum number of parallel workers for processing
        """
        self.llm_model_id = llm_model_id
        self.api_token = api_token
        self.max_workers = max_workers
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200, 
            add_start_index=True
        )
    
    def _get_inference_client(self) -> InferenceClient:
        """Create and return a new inference client."""
        return InferenceClient(self.llm_model_id, token=self.api_token)
    
    def _extract_text_from_file(self, filepath: Path) -> Optional[str]:
        """
        Extract text content from resume file.
        
        Args:
            filepath: Path to the resume file
            
        Returns:
            Extracted text content or None if extraction failed
        """
        try:
            file_extension = filepath.suffix.lower()
            
            if file_extension == ".pdf":
                loader = PyPDFLoader(str(filepath))
            elif file_extension == ".docx":
                loader = Docx2txtLoader(str(filepath))
            else:
                logger.warning(f"Unsupported file type: {file_extension}")
                return None
            
            docs = loader.load()
            
            # Combine all document content
            all_content = ""
            for doc in docs:
                all_content += doc.page_content
            
            return all_content
            
        except Exception as e:
            logger.error(f"Error extracting text from {filepath.name}: {str(e)}")
            return None
    
    def _extract_field_with_llm(
        self, 
        client: InferenceClient, 
        text: str, 
        field_name: str, 
        prompt_template: str
    ) -> str:
        """
        Extract a specific field from resume text using an LLM.
        
        Args:
            client: InferenceClient instance
            text: Resume text
            field_name: Name of the field to extract
            prompt_template: Template for the extraction prompt
            
        Returns:
            Extracted field value
        """
        prompt = prompt_template.format(text=text)
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = client.chat_completion(messages, max_tokens=1000)
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error extracting {field_name}: {str(e)}")
            return f"Error extracting {field_name}"
    
    def extract_info_from_resume(self, filepath: Path) -> Optional[CandidateInfo]:
        """
        Extract structured information from a resume file.
        
        Args:
            filepath: Path to the resume file
            
        Returns:
            CandidateInfo object or None if extraction failed
        """
        text = self._extract_text_from_file(filepath)
        if not text:
            return None
        
        try:
            client = self._get_inference_client()
            
            # Extract various fields
            name = self._extract_field_with_llm(
                client, text, "name",
                "Output only the full name of the candidate in the resume below: {text}"
            )
            
            years_of_experience = self._extract_field_with_llm(
                client, text, "years of experience",
                "Output only the number of years of experience from the following resume, no commentary: {text}"
            )
            
            degree = self._extract_field_with_llm(
                client, text, "degree",
                "Output only the academic degrees earned by the individual in the following resume, no commentary: {text}"
            )
            
            experience_summary = self._extract_field_with_llm(
                client, text, "experience",
                "Summarize this individual's experience and areas of expertise in 3-5 sentences, no commentary: {text}"
            )
            
            return CandidateInfo(
                name=name,
                years_of_experience=years_of_experience,
                degree=degree,
                experience_summary=experience_summary,
                source_file=filepath.name
            )
            
        except Exception as e:
            logger.error(f"Error processing {filepath.name}: {str(e)}")
            return None
    
    def process_directory(self, directory_path: str) -> pd.DataFrame:
        """
        Process all resume files in a directory.
        
        Args:
            directory_path: Path to directory containing resume files
            
        Returns:
            DataFrame with extracted information
        """
        directory = Path(directory_path)
        all_data = []
        
        # Get all PDF and DOCX files in the directory
        files = list(directory.glob("*.pdf")) + list(directory.glob("*.docx"))
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self.extract_info_from_resume, file): file 
                for file in files
            }
            
            for future in as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        all_data.append(result.dict())
                        logger.info(f"Successfully processed {file.name}")
                except Exception as e:
                    logger.error(f"Error processing {file.name}: {str(e)}")
        
        # Convert to DataFrame
        if all_data:
            df = pd.DataFrame(all_data)
            # Rename columns to match existing format
            df = df.rename(columns={
                "name": "Name ",
                "years_of_experience": "Years of experience ",
                "degree": "Degree",
                "experience_summary": "Resume Experience "
            })
            return df
        else:
            logger.warning("No data extracted from resumes")
            return pd.DataFrame()
