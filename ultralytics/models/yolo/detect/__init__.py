# Ultralytics YOLO 🚀, AGPL-3.0 license

from .predict import DetectionPredictor, DetectionPredictor_s2
from .train import DetectionTrainer, DetectionTrainer_s2
from .val import DetectionValidator, DetectionValidator_s2

__all__ = "DetectionPredictor", "DetectionTrainer", "DetectionValidator", "DetectionPredictor_s2", "DetectionTrainer_s2", "DetectionValidator_s2"
