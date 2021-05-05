




from typing import List
from ._transformer import Transformer, ToDense
from ._preprocessor import MRSADatasetPipeline
from sklearn.model_selection import train_test_split


__all__ = ["DataLoader"]


class DataLoader:
    def __init__(self, preprocessed: MRSADatasetPipeline, transformers: List[Transformer, ToDense], shuffle: bool):
        self.dataset      =  preprocessed
        self.transformers = transformers
        self.pipeline     = None
        self.dataset.x_train, self.dataset.x_test, self.dataset.y_train, self.dataset.y_test = train_test_split(self.dataset.x_train, self.dataset.y_train, test_size=0.2, shuffle=shuffle)

    def __call__(self):
        pass

    
    def plot(self):
        # TODO - dataset plotting
        pass

        