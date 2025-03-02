# Resume Matcher Documentation

## Overview

Resume Matcher is a tool for matching candidate resumes with job listings using natural language processing and embedding-based semantic search. The application uses the Hugging Face API to generate embeddings for both candidate profiles and job listings, then calculates similarity scores to identify the best matches.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd resume-matcher

# Install the package in development mode
pip install -e .

# Create .env file with your API token
echo "HUGGINGFACE_TOKEN=your_token_here" > .env
```

## Configuration

The application is configured using a `config.yaml` file at the project root. Example configuration:

```yaml
file_paths:
  candidate_file: "./data/candidate_data-gap-solutions-only.csv"
  listings_file: "./data/t04_itid.csv"
  output_dir: "./output"

hugging_face:
  model_id: "sentence-transformers/all-MiniLM-L6-v2"
  hf_token: "${HUGGINGFACE_TOKEN}"

debug_mode: true
```

## Usage

The application provides a command-line interface with several commands:

### Process Resumes

Process resume files (PDF, DOCX) to extract structured information:

```bash
resume-matcher process ./path/to/resumes --output candidates.csv
```

### Match Candidates to Jobs

Match candidates to job listings based on similarity:

```bash
resume-matcher match --candidates ./data/candidates.csv --jobs ./data/jobs.csv
```

### Run Full Pipeline

Execute the complete workflow from processing resumes to matching:

```bash
resume-matcher pipeline --process-resumes --resume-dir ./data/resumes
```

## Architecture

The application follows a modular design with these key components:

1. **ResumeMatcherApp**: Main application class orchestrating the entire system
2. **ConfigManager**: Handles configuration loading and validation
3. **EmbeddingService**: Generates text embeddings via Hugging Face API
4. **MatchingEngine**: Calculates similarity and matches candidates to jobs
5. **CandidateRanker**: Ranks candidates for each job based on similarity
6. **ResumeExtractor**: Extracts structured information from resume files

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

## Output

The application generates several outputs in the specified output directory:

1. **similarity_matrix.csv**: Full similarity scores between all candidates and jobs
2. **top_matches.csv**: Ranked list of best candidate matches for each job
3. **similarity_table.html**: Interactive table visualization of similarity scores
4. **similarity_heatmap.html**: Heatmap visualization of the similarity matrix

## Troubleshooting

### Authentication Issues

If you encounter a 401 Unauthorized error, check:
- Your Hugging Face token is valid
- The .env file is properly formatted and located in the project root
- The application is correctly loading the environment variables

### Visualization Issues

If visualizations aren't working, ensure you have installed the required dependencies:
```bash
pip install plotly
```

## Extending the Application

To extend the application, you can:

1. Add new extractors for different resume formats
2. Implement additional matching algorithms in the MatchingEngine
3. Create new visualization methods for the results
4. Enhance the ranking system with additional criteria
