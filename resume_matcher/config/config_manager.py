import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FilePaths:
    candidate_file: Path
    listings_file: Path
    output_dir: Path


@dataclass
class HuggingFaceConfig:
    model_id: str
    api_token: str
    api_url: str
    use_local: bool = False


@dataclass
class LLMConfig:
    model_id: str
    api_token: str


@dataclass
class AppConfig:
    file_paths: FilePaths
    huggingface: HuggingFaceConfig
    llm: Optional[LLMConfig] = None
    top_candidates_file: Optional[Path] = None
    debug_mode: bool = False


class ConfigManager:
    """Manages application configuration with environment variable support and validation."""

    def __init__(self, config_path: str = None):
        """
        Initialize the configuration manager.

        Args:
            config_path: Path to the YAML config file. If None, will look for CONFIG_PATH env var.
        """
        self.config_path = config_path or os.environ.get('CONFIG_PATH', 'config.yaml')
        self.config_data = self._load_config()
        self.app_config = self._create_app_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file with environment variable overrides."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)

            # Override with environment variables if they exist
            if 'HUGGINGFACE_TOKEN' in os.environ:
                config['hugging_face']['hf_token'] = os.environ['HUGGINGFACE_TOKEN']

            return config
        except Exception as e:
            raise ValueError(f"Failed to load configuration: {str(e)}")

    def _get_hf_token(self) -> str:
        """Retrieve the Hugging Face API token from config or environment variables."""
        # Try to get the token from the config data
        token = self.config_data.get('hugging_face', {}).get('hf_token')
        if token:
            return token
        # Fall back to environment variable if not in config
        token = os.environ.get('HUGGINGFACE_TOKEN')
        if token:
            return token
        # Raise an error if token is missing
        raise ValueError("Hugging Face API token not found in config or environment variables.")

    def _create_app_config(self) -> AppConfig:
        """Create and validate the application configuration."""
        # Create FilePaths config
        file_paths = FilePaths(
            candidate_file=Path(self.config_data['file_paths']['candidate_file']),
            listings_file=Path(self.config_data['file_paths']['listings_file']),
            output_dir=Path(self.config_data['file_paths']['output_dir'])
        )

        # Get Hugging Face token using the new method
        hf_token = self._get_hf_token()
        use_local = self.config_data['hugging_face'].get('use_local', False)
        huggingface = HuggingFaceConfig(
            model_id=self.config_data['hugging_face']['model_id'],
            api_token=hf_token,
            api_url=f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.config_data['hugging_face']['model_id']}",
            use_local=use_local
        )

        # Create LLM config if it exists in the config data
        llm_config = None
        if 'llm' in self.config_data:
            llm_config = LLMConfig(
                model_id=self.config_data['llm']['model_id'],
                api_token=self._resolve_env_vars(self.config_data['llm']['api_token'])
            )

        # Create AppConfig with all fields
        app_config = AppConfig(
            file_paths=file_paths,
            huggingface=huggingface,
            llm=llm_config,
            debug_mode=self.config_data.get('debug_mode', False)
        )

        # If top_candidates config exists, add it
        if 'top_candidates' in self.config_data:
            app_config.top_candidates_file = Path(self.config_data['top_candidates']['output_file'])

        # Return the fully constructed app_config object
        return app_config

    def _resolve_env_vars(self, value):
        """Resolve environment variables in configuration values."""
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            env_var_name = value[2:-1]
            env_value = os.environ.get(env_var_name)
            if env_value is None:
                raise ValueError(f"Environment variable {env_var_name} not found")
            return env_value
        return value

    def validate_paths(self) -> bool:
        """Validate that all required files and directories exist."""
        # Check input files
        if not self.app_config.file_paths.candidate_file.exists():
            raise FileNotFoundError(f"Candidate file not found: {self.app_config.file_paths.candidate_file}")

        if not self.app_config.file_paths.listings_file.exists():
            raise FileNotFoundError(f"Listings file not found: {self.app_config.file_paths.listings_file}")

        # Ensure output directory exists, create if it doesn’t
        os.makedirs(self.app_config.file_paths.output_dir, exist_ok=True)

        return True

    def get_config(self) -> AppConfig:
        """Get the application configuration."""
        return self.app_config