"""
RM-VMusic Training Package.
"""
from .losses import WeightedCrossEntropyLoss, SupervisedContrastiveLoss, DistributionInvarianceLoss
from .trainer import train_single_model, fit_model, evaluate_model, predict_model, FeatureTensorDataset, compute_class_weights, set_seed

__all__ = [
    "WeightedCrossEntropyLoss",
    "SupervisedContrastiveLoss",
    "DistributionInvarianceLoss",
    "train_single_model",
    "fit_model",
    "evaluate_model",
    "predict_model",
    "FeatureTensorDataset",
    "compute_class_weights",
    "set_seed"
]
