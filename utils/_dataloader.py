



import numpy as np
from typing import List
from ._transformer import Transformer
from ._preprocessor import MRSADatasetPipeline
from sklearn.model_selection import train_test_split
import pickle, os


__all__ = ["DataLoader"]


class DataLoader:
    def __init__(self, dataset: MRSADatasetPipeline, transformers: List[Transformer]):
        self.dataset      = dataset
        self.transformers = transformers
        self.pipeline     = None


    def __call__(self):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        vocab_path = os.path.abspath(curr_dir + f"/../utils/vocabulary/vocab.p")
        # NOTE - method 1 : 
        # pickle.dump(self.dataset.tfidf_vector, open(vocab_path, "wb"))
        # NOTE - method 2 : 
        pickle.dump(self.pipeline[1], open(vocab_path, "wb"))
        

    
    def plot(self):
        # TODO - dataset plotting
        pass

        