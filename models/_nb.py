






import sys
sys.path.append("..")
from utils import DataLoader
from ._base import BaseLine
from sklearn.naive_bayes import MultinomialNB, GaussianNB


__all__ = ["NaiveBayesian"]


class NaiveBayesian(BaseLine):
    def __init__(self, training_dataloader: DataLoader, algorithm: str):
        args = {"dataloader": training_dataloader, "model": MultinomialNB() if algorithm == "multimonial" else GaussianNB()}
        super().__init__(**args)