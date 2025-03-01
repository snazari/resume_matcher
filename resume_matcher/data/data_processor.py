"""
Module for processing and transforming data in the Resume Matcher system.
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import os
import re

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Class for processing and transforming data for the Resume Matcher system.
    """
    
    def __init__(self):
        """Initialize the data processor."""
        pass
    
    def combine_columns(
        self, 
        df: pd.DataFrame, 
        columns: List[str], 
        new_column: str
    ) -> pd.DataFrame:
        """
        Combine multiple columns into a single text column.
        
        Args:
            df: DataFrame to process
            columns: List of column names to combine
            new_column: Name of the new combined column
            
        Returns:
            DataFrame with the new combined column
        """
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Ensure all columns exist, create empty ones if missing
        for col in columns:
            if col not in result_df.columns:
                logger.warning(f"Column '{col}' not found, creating empty column")
                result_df[col] = ""
        
        # Combine columns with space separator
        result_df[new_column] = result_df[columns].astype(str).agg(' '.join, axis=1)
        
        # Remove multiple spaces
        result_df[new_column] = result_df[new_column].apply(
            lambda x: re.sub(r'\s+', ' ', x).strip()
        )
        
        return result_df
    
    def clean_text_data(self, df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
        """
        Clean text data in specified columns.
        
        Args:
            df: DataFrame to process
            text_columns: List of text column names to clean
            
        Returns:
            DataFrame with cleaned text columns
        """
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        for col in text_columns:
            if col in result_df.columns:
                # Convert to string
                result_df[col] = result_df[col].astype(str)
                
                # Remove extra whitespace
                result_df[col] = result_df[col].apply(
                    lambda x: re.sub(r'\s+', ' ', x).strip()
                )
                
                # Replace common characters that might cause issues
                result_df[col] = result_df[col].str.replace('\n', ' ')
                result_df[col] = result_df[col].str.replace('\r', ' ')
                result_df[col] = result_df[col].str.replace('\t', ' ')
                
                # Clean up again
                result_df[col] = result_df[col].apply(
                    lambda x: re.sub(r'\s+', ' ', x).strip()
                )
        
        return result_df
    
    def handle_missing_values(
        self, 
        df: pd.DataFrame, 
        numeric_fill: Union[int, float] = 0, 
        text_fill: str = ""
    ) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame.
        
        Args:
            df: DataFrame to process
            numeric_fill: Value to use for filling numeric columns
            text_fill: Value to use for filling text columns
            
        Returns:
            DataFrame with missing values handled
        """
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Process each column based on its type
        for col in result_df.columns:
            if pd.api.types.is_numeric_dtype(result_df[col]):
                # Fill numeric columns
                result_df[col] = result_df[col].fillna(numeric_fill)
            else:
                # Fill text/object columns
                result_df[col] = result_df[col].fillna(text_fill)
        
        return result_df
    
    def normalize_years_of_experience(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Normalize years of experience column to numeric values.
        
        Args:
            df: DataFrame to process
            column: Column name containing years of experience
            
        Returns:
            DataFrame with normalized years of experience
        """
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        if column not in result_df.columns:
            logger.warning(f"Column '{column}' not found")
            return result_df
        
        # Define patterns to extract numeric values
        patterns = [
            r'(\d+)\s*\+?\s*years',  # "10+ years", "10 years"
            r'(\d+)\s*\+?\s*yrs',     # "10+ yrs", "10 yrs"
            r'(\d+)\s*\+?\s*year',    # "1+ year", "1 year"
            r'(\d+)\s*\+?\s*yr',      # "1+ yr", "1 yr"
            r'(\d+)\s*\+',            # "10+"
            r'^(\d+)$'                # Just a number
        ]
        
        # Function to extract years from text
        def extract_years(text):
            text = str(text).lower().strip()
            
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    return int(match.group(1))
            
            # If no pattern matches, try to directly convert to float
            try:
                return float(text)
            except (ValueError, TypeError):
                return None
        
        # Apply the extraction function
        result_df[f'{column}_numeric'] = result_df[column].apply(extract_years)
        
        # Fill missing values with 0
        result_df[f'{column}_numeric'] = result_df[f'{column}_numeric'].fillna(0)
        
        return result_df
    
    def split_data(
        self, 
        df: pd.DataFrame, 
        column: str, 
        split_pattern: str = ',', 
        strip: bool = True
    ) -> pd.DataFrame:
        """
        Split a column with delimited values into multiple rows.
        
        Args:
            df: DataFrame to process
            column: Column name containing delimited values
            split_pattern: Delimiter pattern for splitting
            strip: Whether to strip whitespace from split values
            
        Returns:
            DataFrame with split values
        """
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        if column not in result_df.columns:
            logger.warning(f"Column '{column}' not found")
            return result_df
        
        # Convert column to string
        result_df[column] = result_df[column].astype(str)
        
        # Function to split values
        def split_values(row):
            values = re.split(split_pattern, row[column])
            if strip:
                values = [v.strip() for v in values]
            
            # Create a new row for each value
            result = []
            for value in values:
                if value:  # Skip empty values
                    new_row = row.copy()
                    new_row[column] = value
                    result.append(new_row)
            
            return result
        
        # Apply the split function and explode the result
        expanded_rows = []
        for _, row in result_df.iterrows():
            expanded_rows.extend(split_values(row))
        
        # Create a new DataFrame from the expanded rows
        if expanded_rows:
            return pd.DataFrame(expanded_rows)
        else:
            return result_df
    
    def add_contingent_status(
        self, 
        df: pd.DataFrame, 
        contingent_df: pd.DataFrame, 
        name_column: str = 'Name '
    ) -> pd.DataFrame:
        """
        Add contingent status information to the main DataFrame.
        
        Args:
            df: Main DataFrame
            contingent_df: DataFrame with contingent information
            name_column: Column name containing candidate names
            
        Returns:
            DataFrame with contingent status added
        """
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        if name_column not in result_df.columns:
            logger.warning(f"Column '{name_column}' not found in main DataFrame")
            return result_df
        
        if contingent_df.empty:
            logger.warning("Contingent DataFrame is empty")
            return result_df
        
        # Get the first column from contingent_df as the name column
        contingent_names = contingent_df.iloc[:, 0].values
        
        # Convert to set for faster lookups
        contingent_set = set(contingent_names)
        
        # Create a boolean column indicating contingent status
        result_df['Contingent'] = result_df[name_column].apply(
            lambda name: name in contingent_set
        )
        
        logger.info(f"Added contingent status: {result_df['Contingent'].sum()} contingent candidates found")
        
        return result_df
