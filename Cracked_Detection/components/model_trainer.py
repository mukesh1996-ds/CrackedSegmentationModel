import os,sys
import yaml
from ultralytics import YOLO
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


    def initiate_model_trainer(self,) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            #logging.info("Unzipping data")
            #os.system("unzip data.zip")
            #os.system("rm data.zip")
            data_yaml_file = "artifacts/data_ingestion/feature_store/data.yaml"

            with open(data_yaml_file, 'r') as stream:
                num_classes = str(yaml.safe_load(stream)['nc'])

            model_config_file_name = self.model_trainer_config.weight_name.split(".")[0]
            print(model_config_file_name)

            #config = read_yaml_file(f"artifacts/data_ingestion/feature_store/{model_config_file_name}.yaml")

            #config['nc'] = int(num_classes)

            model = YOLO('yolov8n-seg.pt')

            results = model.train(data=data_yaml_file,
                      epochs=int(input("Enter the number of epochs you need to train the model:- ")),
                      imgsz=640)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path="yolov8/best.pt",
            )

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact


        except Exception as e:
            raise AppException(e, sys)