

import argparse, sys
from utils import DataLoader, MRSADatasetPipeline, Transformer
from models import LogisticRegression, LinearRegression, SupportVectorMachine, RandomForest, NaiveBayesian


# ------------ processing argument options
# -----------------------------------------------
parser = argparse.ArgumentParser(description='Movie Review Sentiment Analysis')
parser.add_argument('--model', action='store', type=str, help='naive_bayesian, support_vector_machine, random_forest, logistic_regression, linear_regression.', required=True)
parser.add_argument('--train-path', action='store', type=str, help='The training data CSV file path.', required=True)
parser.add_argument('--test-path', action='store', type=str, help='The data CSV file path to test the pre-trained model.', required=True)

args                = parser.parse_args()
model               = args.model
train_path          = args.train_path
test_path           = args.test_path
preprocessed        = MRSADatasetPipeline(paths=[train_path])
training_dataloader = DataLoader(preprocessed=preprocessed, transformers=[Transformer], shuffle=True)
# print(f"[+] preprocessed length : {len(preprocessed)}") # NOTE - testing desing pattern
# print(f"[+] the third sample is : {preprocessed[2]}") # NOTE - testing desing pattern



# ------------ training process based on selected model
# ----------------------------------------------------------
if model == "naive_bayesian":
    model = NaiveBayesian(training_dataloader=training_dataloader)
elif model == "support_vector_machine":
    model = SupportVectorMachine(training_dataloader=training_dataloader)
elif model == "random_forest":
    model = RandomForest(training_dataloader=training_dataloader)
elif model == "logistic_regression":
    model = LogisticRegression(training_dataloader=training_dataloader)
elif model == "linear_regression":
    mode = LinearRegression(training_dataloader=training_dataloader)
else:
    print("[?] Invalid Model!")
    sys.exit(1)
model.train()


# ------------ testing and statistical process based on pre-trained model
# --------------------------------------------------------------------------
statistics = model.stat()
print("\t[+] Accuracy : ", statistics["accuracy"])
print("\t[+] Precision : ", statistics["precision"])
print("\t[+] Recall : ", statistics["recall"])
print("\t[+] f1-score : ", statistics["f1_score"])
predicted  = model([test_path])
print("\t[+] Predicted Label for Test Data : ", predicted)