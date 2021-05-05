






from ._data import Dataset
from typing import List
import pandas as pd
import os, string
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
import spacy
spacy.load('en')


__all__ = ["MRSADatasetPipeline"]


class MRSADatasetPipeline(Dataset):
    def __init__(self, paths: List[str]):
        if not os.path.exists(paths[0]): print("[?] CSV Datasets Not Found!"); sys.exit(1)
        self.__data_path    = f"{paths[0]}/x_train.csv"
        self.__labels_path  = f"{paths[0]}/y_train.csv"
        self.__parser       = English()
        self.__punctuations = string.punctuation
        self.__stop_words   = STOP_WORDS
        self.x_train        = pd.read_csv(self.__data_path)["text"]
        self.y_train        = pd.read_csv(self.__labels_path)["label"] 
        self.x_test         = None
        self.y_test         = None
        self.bag_of_words   = None
        self.tfidf_vector   = None


    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


    def __len__(self):
        return len(self.x_train)


    def __call__(self):
        pass


    def tokenizer(self, doc):
        # NOTE - -PRON- is used as the lemma for all personal pronouns.
        # NOTE - doc might contains multiple sentences which is our training text. 
        tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lemma_ for word in self.__parser(doc)]
        tokens = [word for word in tokens if word not in self.__stop_words and word not in self.__punctuations]
        return tokens
        

    def vectorizer(self):
        # NOTE - in BOW we'll have frequencies, rather than just 1 or 0 for their occurrence in ngram.
        # NOTE - CountVectorizer, HashingVectorizer and TfidfVectorizer will crash on making a huge 
        #        vocabulary dictionary based on a dense matrix in memory when we're trying to 
        #        use naive bayes gaussian algo. cause the inputs of the naive bayes gaussian model 
        #        is a dense matrix and as the name of the algo says it's based on probability distribution 
        #        not a sparse matrix which is the output of vectorization algos.   
        # NOTE - we're using BOW with TF-IDF normalization for building our vocabulary and feature extraction.
        # self.bag_of_words = CountVectorizer(tokenizer=self.tokenizer, ngram_range=(1,1))
        # self.bag_of_words = HashingVectorizer(tokenizer=self.tokenizer, ngram_range=(1,1))
        self.tfidf_vector = TfidfVectorizer(tokenizer=self.tokenizer)
        return self.tfidf_vector


