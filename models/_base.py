


# https://towardsdatascience.com/text-classification-using-naive-bayes-theory-a-working-example-2ef4b7eb7d5a
# https://www.dataquest.io/blog/tutorial-text-classification-in-python-using-spacy/


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


class BaseLine:
    def __init__(self, *args, **kwargs):
        self.dataloader = kwargs["dataloader"]
        self.model      = kwargs["model"]

    def stat(self):
        pass

    def __call__(self):
        pass

    def train(self):
        pipe = self.dataloader(self.model)
        pipe.fit(self.dataloader.x_train, self.dataloader.y_train)