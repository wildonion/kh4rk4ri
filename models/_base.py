




from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from typing import List
import pandas as pd
import os
import seaborn as sns
import logging
logging.basicConfig(level=logging.DEBUG)


class BaseLine:
    def __init__(self, *args, **kwargs):
        self.dataloader = kwargs["dataloader"]
        self.model      = kwargs["model"]
        logging.info('[+] Filling dataloader pipeline with preprocessed inputs...')
        self.dataloader.pipeline = Pipeline([("cleaner", self.dataloader.transformers[0]()),
                                             ('vectorizer', self.dataloader.dataset.vectorizer()),
                                             ('classifier', self.model)]) # filling dataloader piepline with preprocessed inputs

    def stat(self):
        """
                    accuracy: (tp + tn) / (p + n)
                    precision tp / (tp + fp)
                    recall: tp / (tp + fn)
                    f1: 2 tp / (2 tp + fp + fn)
        """
        logging.info('[+] Calculating statistical results...')
        predicted = self.dataloader.pipeline.predict(self.dataloader.dataset.x_test)
        mat = confusion_matrix(self.dataloader.dataset.y_test, predicted)
        sns.heatmap(mat.T, square = True, annot=True, fmt = "d", xticklabels=["Positive Review", "Negative Review"], yticklabels=["Positive Review", "Negative Review"])
        plt.xlabel("true labels")
        plt.ylabel("predicted label")
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        cmat_path = os.path.abspath(curr_dir + f"/../utils/cmat/{self.model.__class__.__name__}.png")
        plt.savefig(cmat_path)
        return {
                "accuracy" : accuracy_score(self.dataloader.dataset.y_test, predicted),
                "precision": precision_score(self.dataloader.dataset.y_test, predicted),
                "recall"   : recall_score(self.dataloader.dataset.y_test, predicted),
                "f1_score" : f1_score(self.dataloader.dataset.y_test, predicted)
                }


    def __call__(self, paths: List[str]):
        logging.info('[+] Labeling test data using dataloader pipeline...')
        if not os.path.exists(paths[0]): print("[?] CSV Test File Not Found!"); sys.exit(1)
        input_test_data = pd.read_csv(f"{paths[0]}/x_test.csv")
        predicted_input_test_data = self.dataloader.pipeline.predict(input_test_data["text"]) # NOTE - all preprocessing is done through the pipeline estimators we've built
        input_test_data['label'] = predicted_input_test_data.tolist()
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        labeled_input_test_data = os.path.abspath(curr_dir + f"/../utils/labeled/{self.model.__class__.__name__}.csv")
        input_test_data.to_csv(labeled_input_test_data, index=False)
        return predicted_input_test_data


    def train(self):
        logging.info('[+] Training through the dataloader pipeline...')
        self.dataloader.pipeline.fit(self.dataloader.dataset.x_train, self.dataloader.dataset.y_train)
        # NOTE - we can access the vocabulary of our tfidf vectorization algo using 
        #        self.dataloader.dataset.tfidf_vector.vocabulary_ or self.dataloader.pipeline[1]
        logging.info('[+] Saving vocabulary at utils/vocabulary/vocab.p')
        self.dataloader() # NOTE - saving tfidf vocabulary for later usage with other models