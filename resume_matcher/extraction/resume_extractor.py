"""
Module for extracting structured information from resume files.
"""
import logging
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import InferenceClient

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

    def __init__(self, model_id: str, api_token: Optional[str] = None, use_local: bool = True, max_workers: int = 4):
        """
        Initialize the resume extractor.

        Args:
            model_id: Identifier for the LLM (local model path/ID or Hugging Face model ID)
            api_token: API token for Hugging Face Inference API (required if use_local is False)
            use_local: Whether to use a local LLM (default True) or the Hugging Face API
            max_workers: Maximum number of parallel workers for processing files
        """
        self.model_id = model_id
        self.api_token = api_token
        self.use_local = use_local
        self.max_workers = max_workers
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )

        if self.use_local:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(model_id)
            self.model.eval()
            logger.info(f"Loaded local model: {model_id}")
        elif not api_token:
            raise ValueError("API token is required when use_local is False")

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
            all_content = "".join(doc.page_content for doc in docs)
            return all_content
        except Exception as e:
            logger.error(f"Error extracting text from {filepath.name}: {str(e)}")
            return None

    def _extract_field_with_llm(self, text: str, field_name: str, prompt_template: str) -> str:
        """
        Extract a specific field from resume text using an LLM, handling long texts by splitting into chunks.

        Args:
            text: Resume text
            field_name: Name of the field to extract
            prompt_template: Template for the extraction prompt

        Returns:
            Extracted field value
        """
        # Split the text into chunks
        chunks = self.text_splitter.split_text(text)
        extracted_parts = []

        for chunk in chunks:
            prompt = prompt_template.format(text=chunk)
            try:
                if self.use_local:
                    # Tokenize with truncation to ensure input fits the model's limit
                    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                    outputs = self.model.generate(**inputs, max_new_tokens=50, do_sample=False)
                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    extracted = response[len(prompt):].strip()
                    if extracted and extracted != f"Error extracting {field_name}":
                        extracted_parts.append(extracted)
                else:
                    client = InferenceClient(self.model_id, token=self.api_token)
                    messages = [{"role": "user", "content": prompt}]
                    response = client.chat_completion(messages, max_tokens=1000)
                    extracted = response.choices[0].message.content.strip()
                    if extracted and extracted != f"Error extracting {field_name}":
                        extracted_parts.append(extracted)
            except Exception as e:
                logger.error(f"Error extracting {field_name} from chunk: {str(e)}")
                extracted_parts.append(f"Error extracting {field_name}")

        # Combine results from all chunks
        if not extracted_parts:
            return f"Error extracting {field_name}"
        # For fields like name or degree, return the first valid result; for summaries, join all parts
        if field_name in ["name", "degree", "years of experience"]:
            return next((part for part in extracted_parts if "Error" not in part), extracted_parts[0])
        return " ".join(extracted_parts)

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
            name = self._extract_field_with_llm(
                text, "name", "Output only the full name of the candidate in the resume below: {text}"
            )
            years_of_experience = self._extract_field_with_llm(
                text, "years of experience",
                "Output only the number of years of experience from the following resume, no commentary: {text}"
            )
            degree = self._extract_field_with_llm(
                text, "degree",
                "Output only the academic degrees earned by the individual in the following resume, no commentary: {text}"
            )
            experience_summary = self._extract_field_with_llm(
                text, "experience",
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
        files = list(directory.glob("*.pdf")) + list(directory.glob("*.docx"))

        # Note: Parallel processing is retained, but local model usage may require thread safety
        # (e.g., threading.Lock around model.generate()) if issues arise.
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {executor.submit(self.extract_info_from_resume, file): file for file in files}
            for future in as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        all_data.append(result.dict())
                        logger.info(f"Successfully processed {file.name}")
                except Exception as e:
                    logger.error(f"Error processing {file.name}: {str(e)}")

        if all_data:
            df = pd.DataFrame(all_data)
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