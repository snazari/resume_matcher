"""
Module for creating visualizations of resume matching results.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

logger = logging.getLogger(__name__)


class Visualizer:
    """Class for creating visualizations of resume matching results."""
    
    def __init__(self):
        """Initialize the visualizer."""
        pass
    
    def create_similarity_heatmap(self, similarity_df: pd.DataFrame) -> go.Figure:
        """
        Create a heatmap visualization of the similarity matrix.
        
        Args:
            similarity_df: DataFrame with similarity scores
            
        Returns:
            Plotly Figure object
        """
        logger.info("Creating similarity heatmap")
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=similarity_df.values.tolist(),
            x=similarity_df.columns.tolist(),
            y=similarity_df.index.tolist(),
            colorscale='RdBu_r',
            hoverongaps=False
        ))
        
        # Update layout
        fig.update_layout(
            title="Candidate-Job Similarity Matrix (Cosine Similarity)",
            xaxis_title="Job Openings",
            yaxis_title="Candidates",
            height=800,
            width=2000,
            font=dict(size=12)
        )
        
        # Improve hover information
        fig.update_traces(
            hovertemplate="Candidate: %{y}<br>Job: %{x}<br>Similarity: %{z:.3f}<extra></extra>"
        )
        
        return fig
    
    def create_candidate_table(self, candidates_df: pd.DataFrame) -> go.Figure:
        """
        Create a table visualization of candidate information.
        
        Args:
            candidates_df: DataFrame with candidate information
            
        Returns:
            Plotly Figure object
        """
        logger.info("Creating candidate table")
        
        # Select columns to display
        display_columns = [
            'Name ', 'Years of experience ', 'Degree', 
            'Resume Experience '
        ]
        
        # Filter to only include columns that exist
        columns = [col for col in display_columns if col in candidates_df.columns]
        
        # Create table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=columns,
                fill_color='lightgrey',
                align='left',
                font=dict(size=14)
            ),
            cells=dict(
                values=[candidates_df[col] for col in columns],
                fill_color='white',
                align='left',
                font=dict(size=12),
                height=30
            )
        )])
        
        # Update layout
        fig.update_layout(
            title="Candidate Information",
            height=600,
            width=2000
        )
        
        return fig
    
    def create_matches_table(self, matches_df: pd.DataFrame) -> go.Figure:
        """
        Create a table visualization of candidate-job matches.
        
        Args:
            matches_df: DataFrame with match information
            
        Returns:
            Plotly Figure object
        """
        logger.info("Creating matches table")
        
        # Ensure the DataFrame has the necessary columns
        required_columns = ['Job', 'Candidate', 'Similarity', 'Rank']
        for col in required_columns:
            if col not in matches_df.columns:
                logger.warning(f"Column '{col}' not found in matches DataFrame")
        
        # Use available columns
        columns = [col for col in required_columns if col in matches_df.columns]
        
        # Create table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=columns,
                fill_color='lightgrey',
                align='left',
                font=dict(size=14)
            ),
            cells=dict(
                values=[matches_df[col] for col in columns],
                fill_color='white',
                align='left',
                font=dict(size=12),
                height=30,
                format=[None, None, '.3f', None]  # Format similarity as 3 decimal places
            )
        )])
        
        # Update layout
        fig.update_layout(
            title="Top Candidate-Job Matches",
            height=600,
            width=2000
        )
        
        return fig
    
    def create_job_category_heatmap(
        self, 
        similarity_df: pd.DataFrame, 
        job_categories: Dict[str, str]
    ) -> go.Figure:
        """
        Create a heatmap grouped by job categories.
        
        Args:
            similarity_df: DataFrame with similarity scores
            job_categories: Dictionary mapping job names to categories
            
        Returns:
            Plotly Figure object
        """
        logger.info("Creating job category heatmap")
        
        # Create a copy of the similarity matrix
        df = similarity_df.copy()
        
        # Add category information
        job_category_list = []
        for job in df.columns:
            category = job_categories.get(job, "Uncategorized")
            job_category_list.append({"Job": job, "Category": category})
        
        # Create category DataFrame
        category_df = pd.DataFrame(job_category_list)
        
        # Sort by category
        category_df = category_df.sort_values("Category")
        
        # Reorder columns in similarity DataFrame
        ordered_jobs = category_df["Job"].tolist()
        df = df[ordered_jobs]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=df.values.tolist(),
            x=df.columns.tolist(),
            y=df.index.tolist(),
            colorscale='RdBu_r',
            hoverongaps=False
        ))
        
        # Add annotations for categories
        current_category = None
        start_x = 0
        
        for i, (_, row) in enumerate(category_df.iterrows()):
            if row["Category"] != current_category:
                if current_category is not None:
                    # Add annotation for previous category
                    fig.add_shape(
                        type="rect",
                        x0=start_x - 0.5,
                        y0=-1.5,
                        x1=i - 0.5,
                        y1=-0.5,
                        line=dict(color="black", width=2),
                        fillcolor="lightgray",
                        opacity=0.7
                    )
                    
                    fig.add_annotation(
                        x=(start_x + i - 1) / 2,
                        y=-1,
                        text=current_category,
                        showarrow=False,
                        font=dict(size=14, color="black")
                    )
                
                current_category = row["Category"]
                start_x = i
        
        # Add annotation for the last category
        if current_category is not None:
            fig.add_shape(
                type="rect",
                x0=start_x - 0.5,
                y0=-1.5,
                x1=len(category_df) - 0.5,
                y1=-0.5,
                line=dict(color="black", width=2),
                fillcolor="lightgray",
                opacity=0.7
            )
            
            fig.add_annotation(
                x=(start_x + len(category_df) - 1) / 2,
                y=-1,
                text=current_category,
                showarrow=False,
                font=dict(size=14, color="black")
            )
        
        # Update layout
        fig.update_layout(
            title="Candidate-Job Similarity by Job Category",
            xaxis_title="Job Openings (Grouped by Category)",
            yaxis_title="Candidates",
            height=1000,
            width=2000,
            margin=dict(t=50, b=100),
            font=dict(size=12)
        )
        
        return fig
    
    def create_dashboard(
        self,
        similarity_df: pd.DataFrame,
        top_matches_df: pd.DataFrame,
        candidates_df: pd.DataFrame = None,
        output_dir: Path = None
    ) -> None:
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            similarity_df: DataFrame with similarity scores
            top_matches_df: DataFrame with top matches
            candidates_df: DataFrame with candidate information
            output_dir: Directory to save the dashboard HTML
            
        Returns:
            None
        """
        logger.info("Creating dashboard")
        
        # Create a dashboard layout
        from plotly.subplots import make_subplots
        
        # Create figures
        heatmap_fig = self.create_similarity_heatmap(similarity_df)
        matches_table_fig = self.create_matches_table(top_matches_df)
        
        # Save individual figures if output_dir is provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            heatmap_fig.write_html(output_dir / "similarity_heatmap.html")
            matches_table_fig.write_html(output_dir / "top_matches_table.html")
            
            if candidates_df is not None:
                candidate_table_fig = self.create_candidate_table(candidates_df)
                candidate_table_fig.write_html(output_dir / "candidate_table.html")
            
            # Create a combined dashboard
            dashboard_html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Resume Matcher Dashboard</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                    .dashboard { display: flex; flex-direction: column; gap: 20px; }
                    .row { display: flex; gap: 20px; }
                    .widget { flex: 1; border: 1px solid #ddd; border-radius: 5px; overflow: hidden; }
                    h1, h2 { text-align: center; }
                    iframe { border: none; width: 100%; height: 100%; }
                </style>
            </head>
            <body>
                <h1>Resume Matcher Dashboard</h1>
                <div class="dashboard">
                    <div class="row" style="height: 600px;">
                        <div class="widget">
                            <h2>Similarity Heatmap</h2>
                            <iframe src="similarity_heatmap.html"></iframe>
                        </div>
                    </div>
                    <div class="row" style="height: 500px;">
                        <div class="widget">
                            <h2>Top Matches</h2>
                            <iframe src="top_matches_table.html"></iframe>
                        </div>
                    </div>
            """
            
            if candidates_df is not None:
                dashboard_html += """
                    <div class="row" style="height: 500px;">
                        <div class="widget">
                            <h2>Candidate Information</h2>
                            <iframe src="candidate_table.html"></iframe>
                        </div>
                    </div>
                """
            
            dashboard_html += """
                </div>
            </body>
            </html>
            """
            
            with open(output_dir / "dashboard.html", "w") as f:
                f.write(dashboard_html)
                
            logger.info(f"Dashboard saved to {output_dir / 'dashboard.html'}")
