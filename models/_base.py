




from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from typing import List
import pandas as pd
import os
import seaborn as sns
import logging
logging.basicConfig(level=logging.DEBUG)
import pickle


# NOTE - this is an abstract class and the self in here is the self from child class cause we're creating an object from the child class not the parent itself.
# NOTE - self in BaseLine class refers to the selected model class itself cause we didn't create any object from the BaseLine class.
# https://stackoverflow.com/questions/25062114/calling-child-class-method-from-parent-class-file-in-python
class BaseLine:
    def __init__(self, *args, **kwargs):
        self.dataloader  = kwargs["dataloader"]
        self.model       = kwargs["model"]
        self.child_class = kwargs["class_name"]
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
        predicted = self.dataloader.pipeline.predict(self.dataloader.dataset.x_valid)
        mat = confusion_matrix(self.dataloader.dataset.y_valid, predicted)
        sns.heatmap(mat.T, square = True, annot=True, fmt = "d", xticklabels=["Positive Review", "Negative Review"], yticklabels=["Positive Review", "Negative Review"])
        plt.xlabel("True Labels")
        plt.ylabel("Predicted Label")
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        cmat_path = os.path.abspath(curr_dir + f"/../utils/cmat/{self.model.__class__.__name__}.png")
        plt.savefig(cmat_path)
        return {
                "accuracy"     : accuracy_score(self.dataloader.dataset.y_valid, predicted),
                "precision"    : precision_score(self.dataloader.dataset.y_valid, predicted),
                "recall"       : recall_score(self.dataloader.dataset.y_valid, predicted),
                "f1_score"     : f1_score(self.dataloader.dataset.y_valid, predicted),
                "roc_auc_score": roc_auc_score(self.dataloader.dataset.y_valid, predicted)
                }


    def __call__(self, paths: List[str]):
        if not os.path.exists(paths[0]): print("[?] CSV Test File Not Found!"); sys.exit(1)
        logging.info('[+] Labeling test data using dataloader pipeline...')
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        input_test_data = pd.read_csv(f"{paths[0]}/x_test.csv")
        predicted_input_test_data = self.dataloader.pipeline.predict(input_test_data["text"]) # NOTE - all preprocessing is done through the pipeline estimators we've built
        
        labeled_input_test_data_path = os.path.abspath(curr_dir + f"/../utils/labeled/{self.model.__class__.__name__}.csv")
        labeled_input_test_data_path_on_dataset_folder = os.path.abspath(curr_dir + f"/../dataset/test/{self.model.__class__.__name__}-y_test.csv")
        
        labeled_input_test_data = pd.DataFrame(data={'id': input_test_data["ID"], 'label': predicted_input_test_data.tolist()})
        labeled_input_test_data.to_csv(labeled_input_test_data_path, index=False)
        labeled_input_test_data.to_csv(labeled_input_test_data_path_on_dataset_folder, index=False)
        logging.info(f'[+] Labeled test data saved at {labeled_input_test_data_path} and {labeled_input_test_data_path_on_dataset_folder}')
        return predicted_input_test_data



    def train(self):
        logging.info('[+] Training through the dataloader pipeline...')
        self.dataloader.pipeline.fit(self.dataloader.dataset.x_train, self.dataloader.dataset.y_train)
        self.__save() # NOTE - will save the whole BaseLine and the selected model class, see the first NOTE above on top of the BaseLine class


    
    def __save(self):
        logging.info(f'[+] Saving BaseLine class at utils/model/BaseLine.bcls')
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.abspath(curr_dir + f"/../utils/model/BaseLine.bcls")
        pickle.dump(self, open(model_path, "wb"))
