



import sys
sys.path.append("..")
from utils import DataLoader
from ._base import BaseLine
from sklearn.linear_model import LogisticRegression


__all__ = ["LogisticRegression"]


class LogisticRegression(BaseLine):
    def __init__(self, training_dataloader: DataLoader):
        args = {"dataloader": training_dataloader, "model": LogisticRegression()}
        super().__init__(**args)