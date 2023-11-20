import locale
from roboflow import Roboflow

from super_gradients.training import models
from super_gradients.training import Trainer
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_val
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

import os
from typing import List, Dict


locale.getpreferredencoding = lambda: "UTF-8"

"""
rf = Roboflow(api_key="VvcEs35JKM2WFJY30ijR")
project = rf.workspace("tesis-8euxs").project("tesis-yolo")
dataset = project.version(1).download("yolov8")
"""



ruta_actual = os.getcwd()

class config:
    # Project paths
    DATA_DIR: str = ruta_actual + "/Tesis-Yolo-1"
    CHECKPOINT_DIR: str = ruta_actual + "/checkpoints"
    EXPERIMENT_NAME: str = "tesis_detection_model_10k_yolo"

    # Datasets
    TRAIN_IMAGES_DIR: str = ruta_actual + "/Tesis-Yolo-1/train/images"
    TRAIN_LABELS_DIR: str = ruta_actual + "/Tesis-Yolo-1/train/labels"
    VAL_IMAGES_DIR: str = ruta_actual + "/Tesis-Yolo-1/valid/images"
    VAL_LABELS_DIR: str = ruta_actual + "/Tesis-Yolo-1/valid/labels"
    TEST_IMAGES_DIR: str = ruta_actual + "/Tesis-Yolo-1/test/images"
    TEST_LABEL_DIR: str = ruta_actual + "/Tesis-Yolo-1/test/labels"

    # Classes
    CLASSES: List[str] = ['Persona','arma']
    NUM_CLASSES: int = len(CLASSES)

    # Model
    DATALOADER_PARAMS: Dict = {
    'batch_size': 5,
    'num_workers': 1
    }
    MODEL_NAME: str = 'yolo_nas_l'
    PRETRAINED_WEIGHTS: str = 'coco'



def main():

    train_data = coco_detection_yolo_format_train(
        dataset_params={
            'data_dir': config.DATA_DIR,
            'images_dir': config.TRAIN_IMAGES_DIR,
            'labels_dir': config.TRAIN_LABELS_DIR,
            'classes': config.CLASSES
        },
        dataloader_params=config.DATALOADER_PARAMS
    )

    test_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': config.DATA_DIR,
            'images_dir': config.VAL_IMAGES_DIR,
            'labels_dir': config.VAL_LABELS_DIR,
            'classes': config.CLASSES
        },
        dataloader_params=config.DATALOADER_PARAMS
    )

    val_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': config.DATA_DIR,
            'images_dir': config.VAL_IMAGES_DIR,
            'labels_dir': config.VAL_LABELS_DIR,
            'classes': config.CLASSES
        },
        dataloader_params=config.DATALOADER_PARAMS
    )

    train_params = {
        "average_best_models":True,
        "warmup_mode": "linear_epoch_step",
        "warmup_initial_lr": 1e-6,
        "lr_warmup_epochs": 1, #se cambio de 1 a 3 epocas
        "initial_lr": 5e-4,
        "lr_mode": "cosine",
        "cosine_final_lr_ratio": 0.1,
        "optimizer": "Adam",
        "optimizer_params": {"weight_decay": 0.001},
        "zero_weight_decay_on_bias_and_bn": True,
        "ema": True,
        "ema_params": {"decay": 0.9, "decay_type": "threshold"},
        "max_epochs": 40,
        "mixed_precision": True,
        "loss": PPYoloELoss(
            use_static_assigner=False,
            num_classes=config.NUM_CLASSES,
            reg_max=16
        ),
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres=0.1,
                top_k_predictions=300, #
                num_cls=config.NUM_CLASSES,
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.4
                )
            )
        ],
        "metric_to_watch": 'mAP@0.50'
    }

    model = models.get(config.MODEL_NAME, num_classes=config.NUM_CLASSES, pretrained_weights=config.PRETRAINED_WEIGHTS)

    trainer = Trainer(experiment_name=config.EXPERIMENT_NAME, ckpt_root_dir=config.CHECKPOINT_DIR)

    trainer.train(model=model, training_params=train_params, train_loader=train_data, valid_loader=val_data)

    best_model = models.get(config.MODEL_NAME,
                            num_classes=config.NUM_CLASSES,
                            checkpoint_path=os.path.join(config.CHECKPOINT_DIR, config.EXPERIMENT_NAME, ruta_actual +'/checkpoints/tesis_detection_model_10k_yolo/average_model.pth'))

    return best_model

main()