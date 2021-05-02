





import sys
sys.path.append("..")
from utils import DataLoader
from ._base import BaseLine
from sklearn.svm import SVC


__all__ = ["SupportVectorMachine"]


class SupportVectorMachine(BaseLine):
    def __init__(self, training_dataloader: DataLoader):
        args = {"dataloader": training_dataloader, "model": SVC()}
        super().__init__(**args)