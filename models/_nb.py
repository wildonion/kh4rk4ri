






import sys
sys.path.append("..")
from utils import DataLoader
from ._base import BaseLine
from sklearn.naive_bayes import MultinomialNB


__all__ = ["NaiveBayesian"]


class NaiveBayesian(BaseLine):
    def __init__(self, training_dataloader: DataLoader):
        args = {"dataloader": training_dataloader, "model": MultinomialNB()}
        super().__init__(**args)