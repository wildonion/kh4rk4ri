





class BaseLine:
    def __init__(self, *args, **kwargs):
        self.dataloader = kwargs["dataloader"]

    def stat(self):
        # TODO - classification confusion matrix and scores
        pass

    def __call__(self):
        pass