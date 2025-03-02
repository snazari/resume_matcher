"""
Setup script for the Resume Matcher package.
"""
from setuptools import setup, find_packages

# Get version from package
with open("resume_matcher\__init__.py", "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break

# Read README for long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="resume-matcher",
    version=version,
    description="Match candidate resumes with job openings using NLP and embeddings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Sam Nazari, Ph.D.",
    author_email="sam@cognym.ai",
    url="https://github.com/yourusername/resume-matcher",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "plotly>=5.3.0",
        "huggingface-hub>=0.12.0",
        "langchain>=0.0.200",
        "langchain-community>=0.0.12",
        "langchain-text-splitters>=0.0.1",
        "pydantic>=1.9.0",
        "pyyaml>=6.0",
        "requests>=2.27.0",
        "backoff>=2.0.0",
        "python-docx>=0.8.11",
        "python-dotenv>=0.19.0",
        "PyPDF2>=2.0.0",
        "faiss-cpu>=1.7.2",  # Added FAISS for vector database
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.1.0",
            "flake8>=4.0.0",
            "mypy>=0.930",
            "isort>=5.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "resume-matcher=resume_matcher.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Human Resources",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Office Suites",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)