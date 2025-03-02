# Resume Matcher

A modern system for matching candidate resumes with job listings using natural language processing and embedding-based semantic search.

## Features

- Extract structured information from resume files (PDF, DOCX)
- Generate embeddings for candidate profiles and job listings
- Calculate similarity scores to find the best matches
- Visualize results with interactive heatmaps and tables
- Command-line interface for easy use in workflows
- Export results to CSV for further analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/resume-matcher.git
cd resume-matcher

# Install the package
pip install -e .

# Copy the environment template and configure your variables
cp .env.template .env
# Edit .env with your API keys and paths
```

## Configuration

Create a `config.yaml` file with your settings:

```yaml
file_paths:
  candidate_file: "./data/candidates.csv"
  listings_file: "./data/jobs.csv"
  output_dir: "./output" 

hugging_face:
  model_id: "sentence-transformers/all-MiniLM-L6-v2"
  hf_token: "${HUGGINGFACE_TOKEN}"

llm:
  model_id: "meta-llama/Llama-3.3-70B-Instruct"
  api_token: "${LLM_TOKEN}"

debug_mode: false
```

## Usage

### Command Line

```bash
# Process resume files
resume-matcher process ./data/resumes --output ./data/candidates.csv

# Match candidates to job listings
resume-matcher match --candidates ./data/candidates.csv --jobs ./data/jobs.csv

# Run the full pipeline
resume-matcher pipeline --process-resumes --resume-dir ./data/resumes
```

### Python API

```python
from resume_matcher import ResumeMatcherApp

# Initialize the application
app = ResumeMatcherApp(config_path="config.yaml")

# Run the full pipeline
results = app.run_full_pipeline(
    process_resumes=True,
    resume_dir="./data/resumes",
    visualize=True
)

# Get the top matches
top_matches = results["top_matches"]
print(top_matches.head(10))
```

## Data Format

### Candidate CSV Format

```
Name,Years of experience,Degree,Resume Experience
John Doe,5,Bachelor of Computer Science,Software development with Python...
```

### Job Listing CSV Format

```
Role,Description,Degree,Years of Exp,Expanded Experience,Notes
Data Scientist,Developing machine learning models...,Master's or PhD,3-5,Python experience...,Remote
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black resume_matcher tests
```

## TODO

I've built a system that extracts information from resumes, generates embeddings for candidates and job listings, and 
calculates similarity scores to match candidates with positions. Here are recommendations for improving the codebase:

1. **Testing Framework**: 
   - Add unit tests for each module (pytest recommended)
   - Create integration tests for the full pipeline
   - Include test data fixtures

2. **Dependency Management**:
   - Use a `requirements.txt` or `pyproject.toml` file
   - Consider using a virtual environment or Docker containerization

3. **API Improvements**:
   - Add API rate limiting and caching
   - Consider implementing a local embedding model option to reduce API costs

4. **ML Enhancements**:
   - Explore more advanced embedding models (e.g., E5, SBERT)
   - Consider using fine-tuned models for the domain
   - Add model evaluation metrics to compare different approaches

5. **User Interface**:
   - Create a simple web UI using Streamlit or Flask
   - Add interactive visualizations with Plotly

6. **Pipeline Orchestration**:
   - Consider using Airflow or Luigi for complex pipelines
   - Add monitoring and alerting capabilities

## 8. Migration Plan

1. **Phase 1: Setup and Infrastructure**
   - Create the package structure
   - Set up configuration management
   - Implement logging framework

2. **Phase 2: Core Components**
   - Implement EmbeddingService with improved error handling
   - Develop the ResumeExtractor with parallel processing
   - Create the MatchingEngine with visualization capabilities

3. **Phase 3: Integration and Testing**
   - Implement the main application class
   - Write unit and integration tests
   - Test with sample data

4. **Phase 4: Enhancements**
   - Add performance optimizations
   - Implement advanced ML features
   - Create web UI (if needed)
