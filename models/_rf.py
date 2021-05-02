




import sys
sys.path.append("..")
from utils import DataLoader
from ._base import BaseLine
from sklearn.ensemble import RandomForestClassifier


__all__ = ["RandomForest"]


class RandomForest(BaseLine):
    def __init__(self, training_dataloader: DataLoader):
        args = {"dataloader": training_dataloader, "model": RandomForestClassifier()}
        super().__init__(**args)