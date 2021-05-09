

import argparse, sys, os
from utils import DataLoader, MRSADatasetPipeline, Transformer
from models import _LogisticRegression, SupportVectorMachine, RandomForest, NaiveBayesian
import logging, pickle
logging.basicConfig(level=logging.DEBUG)


# ------------ processing argument options
# -----------------------------------------------
parser = argparse.ArgumentParser(description='Movie Review Sentiment Analysis')
parser.add_argument('--model', action='store', type=str, help='naive_bayesian, support_vector_machine, random_forest, logistic_regression, linear_regression.', required=True)
parser.add_argument('--train-path', action='store', type=str, help='The training data CSV file path.', required=True)
parser.add_argument('--valid-path', action='store', type=str, help='The valid data CSV file path.', required=True)
parser.add_argument('--test-path', action='store', type=str, help='The data CSV file path to test the pre-trained model.', required=True)

args                = parser.parse_args()
model               = args.model
train_path          = args.train_path
test_path           = args.test_path
valid_path          = args.valid_path
dataset             = MRSADatasetPipeline(paths=[train_path, valid_path])
training_dataloader = DataLoader(dataset=dataset, transformers=[Transformer])
# print(f"[+] preprocessed length : {len(preprocessed)}") # NOTE - testing desing pattern
# print(f"[+] the third sample is : {preprocessed[2]}") # NOTE - testing desing pattern



# ------------ training process based on selected model
# -------------------------------------------------------------------------
if os.path.exists('utils/model/BaseLine.bcls'):
    logging.info('[+] Loading last saved BaseLine class. Delete it if you want to train a new model!')
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    BaseLine_path = os.path.abspath(curr_dir + f"/utils/model/BaseLine.bcls")
    BaseLine_file = open(BaseLine_path, 'rb')
    model = pickle.load(BaseLine_file)
else:
    logging.info('[+] Training on selected model...')
    if model == "naive_bayesian":
        model = NaiveBayesian(training_dataloader=training_dataloader)
    elif model == "support_vector_machine":
        model = SupportVectorMachine(training_dataloader=training_dataloader)
    elif model == "random_forest":
        model = RandomForest(training_dataloader=training_dataloader)
    elif model == "logistic_regression":
        model = _LogisticRegression(training_dataloader=training_dataloader)
    else:
        print("[?] Invalid Model!")
        sys.exit(1)
    model.train()


# ------------ testing and statistical process based on pre-trained model
# --------------------------------------------------------------------------
statistics = model.stat()
print("\t- Accuracy : ", statistics["accuracy"])
print("\t- Precision : ", statistics["precision"])
print("\t- Recall : ", statistics["recall"])
print("\t- f1-score : ", statistics["f1_score"])
print("\t- ROC AUC score : ", statistics["roc_auc_score"])
print("\t- TN : ", statistics["cmat"]["TN"])
print("\t- FP : ", statistics["cmat"]["FP"])
print("\t- FN : ", statistics["cmat"]["FN"])
print("\t- TP : ", statistics["cmat"]["TP"])
predicted  = model([test_path])


# NOTE - testing design pattern
# model.dataloader.pipeline[1].vocabulary_
# model.dataloader.dataset.tfidf_vector.vocabulary_