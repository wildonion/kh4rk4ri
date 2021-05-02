






from typing import List
import pandas as pd
import os, random, string
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
spacy.load('en')



__all__ = ["Preprocessor"]


class Preprocessor:
    class Transformer(TransformerMixin):
        def transform(self, doc, **transform_params):
            return [self.clean_text(text) for text in doc]

        def fit(self, doc, y=None, **fit_params):
            return self

        def get_params(self, deep=True):
            return {}

        def clean_text(self, text):
            return text.strip().lower()


    def __init__(self, paths: List[str]):
        if not os.path.exists(paths[0]): print("[?] CSV Datasets Not Found!"); sys.exit(1)
        self.__data_path    = f"{paths[0]}/x_train.csv"
        self.__labels_path  = f"{paths[0]}/y_train.csv"
        self.__parser       = English()
        self.__punctuations = string.punctuation
        self.__stop_words   = STOP_WORDS
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(pd.read_csv(self.__data_path)["text"], 
                                                                                pd.read_csv(self.__labels_path)["label"], 
                                                                                test_size=0.2)


    def __call__(self):
        # TODO - plot the tokenized data
        return self


    def tokenizer(self, doc):
        # NOTE - -PRON- is used as the lemma for all personal pronouns.
        # NOTE - doc might contains multiple sentences which is our training text. 
        tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lemma_ for word in self.__parser(doc)]
        tokens = [word for word in tokens if word not in self.__stop_words and word not in self.__punctuations]
        return tokens
        

