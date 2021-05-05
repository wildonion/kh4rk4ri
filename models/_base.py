


# https://towardsdatascience.com/text-classification-using-naive-bayes-theory-a-working-example-2ef4b7eb7d5a
# https://www.dataquest.io/blog/tutorial-text-classification-in-python-using-spacy/


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline


class BaseLine:
    def __init__(self, *args, **kwargs):
        self.dataloader = kwargs["dataloader"]
        self.model      = kwargs["model"]
        self.dataloader.pipeline = Pipeline([("cleaner", self.dataloader.transformers[0]()),
                                             ('vectorizer', self.dataloader.dataset.vectorizer()),
                                             ('to_dense', self.dataloader.transformers[1]()),
                                             ('classifier', self.model)])

    def stat(self):
        # TODO - scores and plotting
        pass


    def __call__(self):
        pass


    def train(self):
        self.dataloader.pipeline.fit(self.dataloader.dataset.x_train, self.dataloader.dataset.y_train)