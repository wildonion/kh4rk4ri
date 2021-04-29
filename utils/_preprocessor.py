



import pandas as pd
import numpy as np
import os

__all__ = ["Preprocessor"]


# TODO - tokenizing, padding, cross-validating, tfidf, word2vec and glove

class Preprocessor:
    def __init__(self, paths: List[str]):
        if not os.path.exists(paths[0]) or not os.path.exists(paths[1]): print("[?] CSV Dataset Not Found!"); sys.exit(1)
        self.data_path   = paths[0]
        self.labels_path = paths[1]
        self.x_train     = None
        self.y_train     = None

    def __call__(self):
        pass