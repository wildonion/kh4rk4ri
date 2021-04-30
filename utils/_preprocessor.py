



from typing import List
import pandas as pd
import numpy as np
import os, random
import string
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English


__all__ = ["Preprocessor"]


class Preprocessor:
    def __init__(self, paths: List[str]):
        if not os.path.exists(paths[0]): print("[?] CSV Datasets Not Found!"); sys.exit(1)
        self.__data_path   = f"{paths[0]}/x_train.csv"
        self.__labels_path = f"{paths[0]}/y_train.csv"
        self.x_train       = pd.read_csv(self.__data_path)
        self.y_train       = pd.read_csv(self.__labels_path)
        self.reviews       = []
        self.__parser      = English()
        # print(self.x_train.head())
        # print(self.y_train.head())
        # NOTE - in BOW we'll have frequencies, rather than just 1 or 0 for their occurrence in ngram.
        # NOTE - we're using BOW with TF-IDF normalization for building our vocabulary and feature extraction.
        self.__punctuations = string.punctuation
        self.__en_nlp = spacy.load('en')
        self.__stop_words = STOP_WORDS



    def __call__(self):
        # TODO - plot the tokenized data
        pass


    def tokenizer(self, doc):
        # NOTE - -PRON- is used as the lemma for all personal pronouns.
        # NOTE - doc might contains multiple sentences which is our training text. 
        tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lemma_ for word in self.__parser(doc)]
        tokens = [word for word in tokens if word not in self.__stop_words and word not in self.__punctuations]
        return tokens
        