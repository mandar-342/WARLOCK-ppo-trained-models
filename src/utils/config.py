import yaml
from pathlib import Path

root = Path(__file__).parent.parent.parent

config_path = root / "config.yaml"

with open(config_path, "r") as file:
    config = yaml.safe_load(file)