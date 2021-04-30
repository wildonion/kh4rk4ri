




import sys
sys.path.append("..")
from utils import DataLoader
from ._base import BaseLine

__all__ = ["RandomForest"]


class RandomForest(BaseLine):
    def __init__(self, training_dataloader: DataLoader):
        args = {"dataloader": DataLoader}
        super().__init__(**args)