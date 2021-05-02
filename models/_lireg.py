





import sys
sys.path.append("..")
from utils import DataLoader
from ._base import BaseLine
from sklearn.linear_model import LinearRegression


__all__ = ["LinearRegression"]


class LinearRegression(BaseLine):
    def __init__(self, training_dataloader: DataLoader):
        args = {"dataloader": training_dataloader, "model": LinearRegression()}
        super().__init__(**args)