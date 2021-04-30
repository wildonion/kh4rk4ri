



from typing import List
import pandas as pd
import numpy as np
import os, random
import string
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline


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
        spacy.load('en')
        self.__stop_words = STOP_WORDS
        bag_of_words = CountVectorizer(tokenizer=self.__tokenizer, ngram_range=(1,1))
        tfidf_vector = TfidfVectorizer(tokenizer=self.__tokenizer)


    def __call__(self):
        # TODO - plot the tokenized data
        return self


    def __tokenizer(self, doc):
        # NOTE - -PRON- is used as the lemma for all personal pronouns.
        # NOTE - doc might contains multiple sentences which is our training text. 
        tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lemma_ for word in self.__parser(doc)]
        tokens = [word for word in tokens if word not in self.__stop_words and word not in self.__punctuations]
        return tokens
        

class CustomTransformer(TransformerMixin):
    def transform(self, doc, **transform_params):
        return [self.clean_text(text) for text in doc]

    def fit(self, doc, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

    def clean_text(self, text):
        return text.strip().lower()