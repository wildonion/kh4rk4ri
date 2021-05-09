



import numpy as np
from typing import List
from ._transformer import Transformer
from ._preprocessor import MRSADatasetPipeline


__all__ = ["DataLoader"]


class DataLoader:
    def __init__(self, dataset: MRSADatasetPipeline, transformers: List[Transformer]):
        self.dataset      = dataset
        self.transformers = transformers
        self.pipeline     = None


    def __call__(self):
        pass

        