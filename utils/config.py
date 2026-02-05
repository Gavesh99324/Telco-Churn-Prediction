"""Configuration loader"""
import yaml


class Config:
    """Load and manage configuration"""

    @staticmethod
    def load_config(config_path='config.yaml'):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
