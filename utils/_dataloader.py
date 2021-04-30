



import pandas as pd
import numpy as np
from ._preprocessor import Preprocessor


__all__ = ["DataLoader"]

class DataLoader:
    def __init__(self, preprocessed: Preprocessor):
        # TODO - split: float = 0.8, limit: int = 0
        self.__preprocessed = preprocessed
        self.x_train        = self.__preprocessed.x_train
        self.y_train        = self.__preprocessed.y_train

    def __call__(self):
        pass