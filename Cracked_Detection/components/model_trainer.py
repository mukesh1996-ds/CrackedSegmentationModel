import os,sys
import yaml
from Cracked_Detection.utils.main_utils import read_yaml_file
from Cracked_Detection.logger import logging
from Cracked_Detection.exception import AppException
from Cracked_Detection.entity.config_entity import ModelTrainerConfig
from Cracked_Detection.entity.artifacts_entity import ModelTrainerArtifact

class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.model_trainer_config = model_trainer_config
