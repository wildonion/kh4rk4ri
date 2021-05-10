








import sys
sys.path.append("..")
from utils import DataLoader
from ._base import BaseLine
from sklearn.linear_model import SGDClassifier


__all__ = ["_SGDClassifier"]


class _SGDClassifier(BaseLine):
    def __init__(self, training_dataloader: DataLoader):
        args = {"dataloader": training_dataloader, "model": SGDClassifier(max_iter=1000, penalty='elasticnet', loss='log', alpha=0.00001), "class_name": self.__class__.__name__}
        super().__init__(**args)