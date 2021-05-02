




from sklearn.pipeline import Pipeline

class PipeLine:
    def __init__(self, *args, **kwargs):
        self.preprocessed = kwargs["preprocessed"]
        self.transformer  = kwargs["transformer"]

    
    def build(self, classifier):
        pipe = Pipeline([("cleaner", self.transformer()),
                         ('vectorizer', self.preprocessed.vectorizer()),
                         ('classifier', classifier)])
        return pipe
        
