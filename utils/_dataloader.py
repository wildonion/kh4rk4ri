





from ._preprocessor import Preprocessor, Transformer
from ._pipeline import PipeLine


__all__ = ["DataLoader"]

class DataLoader(PipeLine):
    def __init__(self, preprocessed: Preprocessor, transformer: Transformer):
        args = {"preprocessed": preprocessed, "transformer": transformer}
        super().__init__(**args)


    def __call__(self, classifier):
        return self.build(classifier)