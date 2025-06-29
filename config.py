"""
Configuration management for the pitch deck pipeline.
"""
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class PDFProcessingConfig:
    """Configuration for PDF processing."""
    target_size: tuple = (1500, 1500)
    dpi: int = 200
    output_format: str = "PNG"
    quality: int = 95


@dataclass
class GeminiConfig:
    """Configuration for Gemini API."""
    model_name: str = "gemini-2.0-flash-exp"
    temperature: float = 0.1
    top_p: float = 0.8
    top_k: int = 32
    max_output_tokens: int = 8192
    timeout: int = 300


@dataclass
class LangfuseConfig:
    """Configuration for Langfuse tracking."""
    host: str = "https://cloud.langfuse.com"
    flush_interval: int = 10
    enable_debug: bool = False


@dataclass
class DatasetConfig:
    """Configuration for dataset preparation."""
    test_size: float = 0.2
    min_instruction_length: int = 10
    max_instruction_length: int = 512
    max_response_length: int = 256
    include_slide_context: bool = True


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning."""
    model_name: str = "microsoft/git-base"
    learning_rate: float = 5e-5
    batch_size: int = 4
    num_epochs: int = 3
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 250
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    dataloader_pin_memory: bool = False


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""
    # Directories
    pdf_dir: str = "PitchDecks"
    output_dir: str = "processed_images"
    training_data_dir: str = "training_data"
    models_dir: str = "models"
    logs_dir: str = "logs"
    
    # Processing limits
    max_pdfs: Optional[int] = None
    max_slides_per_pdf: Optional[int] = None
    
    # Component configurations
    pdf_processing: PDFProcessingConfig = field(default_factory=PDFProcessingConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    langfuse: LangfuseConfig = field(default_factory=LangfuseConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    finetuning: FineTuningConfig = field(default_factory=FineTuningConfig)
    
    # Pipeline options
    skip_existing_images: bool = True
    skip_existing_analysis: bool = True
    validate_schemas: bool = True
    create_backup: bool = True
    
    # Retry settings
    max_retries: int = 3
    retry_delay: int = 5
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure directories exist
        for dir_attr in ['pdf_dir', 'output_dir', 'training_data_dir', 'models_dir', 'logs_dir']:
            dir_path = Path(getattr(self, dir_attr))
            dir_path.mkdir(exist_ok=True)
        
        # Validate ranges
        assert 0 < self.dataset.test_size < 1, "test_size must be between 0 and 1"
        assert self.finetuning.learning_rate > 0, "learning_rate must be positive"
        assert self.finetuning.batch_size > 0, "batch_size must be positive"
        assert self.finetuning.num_epochs > 0, "num_epochs must be positive"
    
    @classmethod
    def from_file(cls, config_path: str) -> 'PipelineConfig':
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Create nested configurations
        config = cls()
        
        for key, value in config_dict.items():
            if hasattr(config, key):
                if key in ['pdf_processing', 'gemini', 'langfuse', 'dataset', 'finetuning']:
                    # Handle nested configurations
                    nested_config = getattr(config, key)
                    for nested_key, nested_value in value.items():
                        if hasattr(nested_config, nested_key):
                            setattr(nested_config, nested_key, nested_value)
                else:
                    setattr(config, key, value)
        
        return config
    
    def save_to_file(self, config_path: str):
        """Save configuration to JSON file."""
        config_dict = {}
        
        for key, value in self.__dict__.items():
            if key in ['pdf_processing', 'gemini', 'langfuse', 'dataset', 'finetuning']:
                # Handle nested configurations
                config_dict[key] = value.__dict__
            else:
                config_dict[key] = value
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    def get_environment_overrides(self) -> Dict[str, Any]:
        """Get configuration overrides from environment variables."""
        overrides = {}
        
        # PDF processing overrides
        if os.getenv('PDF_DPI'):
            overrides['pdf_processing.dpi'] = int(os.getenv('PDF_DPI'))
        
        # Gemini overrides
        if os.getenv('GEMINI_MODEL'):
            overrides['gemini.model_name'] = os.getenv('GEMINI_MODEL')
        if os.getenv('GEMINI_TEMPERATURE'):
            overrides['gemini.temperature'] = float(os.getenv('GEMINI_TEMPERATURE'))
        
        # Fine-tuning overrides
        if os.getenv('FINETUNING_MODEL'):
            overrides['finetuning.model_name'] = os.getenv('FINETUNING_MODEL')
        if os.getenv('LEARNING_RATE'):
            overrides['finetuning.learning_rate'] = float(os.getenv('LEARNING_RATE'))
        if os.getenv('BATCH_SIZE'):
            overrides['finetuning.batch_size'] = int(os.getenv('BATCH_SIZE'))
        if os.getenv('NUM_EPOCHS'):
            overrides['finetuning.num_epochs'] = int(os.getenv('NUM_EPOCHS'))
        if os.getenv('USE_LORA'):
            overrides['finetuning.use_lora'] = os.getenv('USE_LORA').lower() == 'true'
        
        # Pipeline overrides
        if os.getenv('MAX_PDFS'):
            overrides['max_pdfs'] = int(os.getenv('MAX_PDFS'))
        
        return overrides
    
    def apply_overrides(self, overrides: Dict[str, Any]):
        """Apply configuration overrides."""
        for key, value in overrides.items():
            if '.' in key:
                # Handle nested configurations
                main_key, sub_key = key.split('.', 1)
                if hasattr(self, main_key):
                    nested_config = getattr(self, main_key)
                    if hasattr(nested_config, sub_key):
                        setattr(nested_config, sub_key, value)
            else:
                if hasattr(self, key):
                    setattr(self, key, value)


def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    """
    Load pipeline configuration with environment variable overrides.
    
    Args:
        config_path: Path to configuration file (optional)
        
    Returns:
        PipelineConfig instance
    """
    if config_path and Path(config_path).exists():
        config = PipelineConfig.from_file(config_path)
    else:
        config = PipelineConfig()
    
    # Apply environment variable overrides
    overrides = config.get_environment_overrides()
    config.apply_overrides(overrides)
    
    return config


def create_default_config(config_path: str = "config.json"):
    """Create a default configuration file."""
    config = PipelineConfig()
    config.save_to_file(config_path)
    print(f"Default configuration saved to {config_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration management")
    parser.add_argument("--create-default", action="store_true", help="Create default config file")
    parser.add_argument("--config-path", default="config.json", help="Path to config file")
    
    args = parser.parse_args()
    
    if args.create_default:
        create_default_config(args.config_path)
    else:
        config = load_config(args.config_path)
        print("Current configuration:")
        config.save_to_file("/tmp/current_config.json")
        with open("/tmp/current_config.json", 'r') as f:
            print(f.read())
