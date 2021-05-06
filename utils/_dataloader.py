




from typing import List
from ._transformer import Transformer
from ._preprocessor import MRSADatasetPipeline
from sklearn.model_selection import train_test_split
import pickle, os


__all__ = ["DataLoader"]


class DataLoader:
    def __init__(self, preprocessed: MRSADatasetPipeline, transformers: List[Transformer], shuffle: bool):
        self.dataset      =  preprocessed
        self.transformers = transformers
        self.pipeline     = None
        self.dataset.x_train, self.dataset.x_test, self.dataset.y_train, self.dataset.y_test = train_test_split(self.dataset.x_train, self.dataset.y_train, test_size=0.2, shuffle=shuffle)

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

        