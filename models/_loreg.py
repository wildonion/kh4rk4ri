



import sys
sys.path.append("..")
from utils import DataLoader
from ._base import BaseLine
from sklearn.linear_model import LogisticRegression


__all__ = ["_LogisticRegression"]


class _LogisticRegression(BaseLine):
    def __init__(self, training_dataloader: DataLoader):
        args = {"dataloader": training_dataloader, "model": LogisticRegression(), "class_name": self.__class__.__name__}
        super().__init__(**args)