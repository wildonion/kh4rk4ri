





from ._preprocessor import Preprocessor
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


__all__ = ["DataLoader"]

class DataLoader:
    def __init__(self, preprocessed: Preprocessor, transformer: Preprocessor.Transformer):
        self.preprocessed = preprocessed
        self.transformer  = transformer


    def __call__(self, classifier):
        pipe = Pipeline([("cleaner", self.transformer()),
                         ('vectorizer', self.vectorizer()),
                         ('classifier', classifier)])
        return pipe


    def vectorizer(self):
        # NOTE - in BOW we'll have frequencies, rather than just 1 or 0 for their occurrence in ngram.
        self.bag_of_words = CountVectorizer(tokenizer=self.preprocessed.tokenizer, ngram_range=(1,1))
        # NOTE - we're using BOW with TF-IDF normalization for building our vocabulary and feature extraction.
        self.tfidf_vector = TfidfVectorizer(tokenizer=self.preprocessed.tokenizer)
        return self.tfidf_vector