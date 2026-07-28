from src.data_manager import data_pipeline
from src.features import generate_multiasset_features
from src.utils import config, root

data_pipeline()
generate_multiasset_features(symbols=config["data"]["symbols"])
