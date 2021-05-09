





import sys
sys.path.append("..")
from utils import DataLoader
from ._base import BaseLine
from sklearn.svm import SVC


__all__ = ["SupportVectorMachine"]


class SupportVectorMachine(BaseLine):
    def __init__(self, training_dataloader: DataLoader):
        args = {"dataloader": training_dataloader, "model": SVC(kernel='rbf', C=1E5), "class_name": self.__class__.__name__} # NOTE - https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
        super().__init__(**args)