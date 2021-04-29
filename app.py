


'''
https://realpython.com/sentiment-analysis-python/
https://towardsdatascience.com/sentiment-analysis-a-how-to-guide-with-movie-reviews-9ae335e6bcb2
https://machinelearningmastery.com/prepare-movie-review-data-sentiment-analysis/

TODO - classification confusion matrix and scores
'''


import argparse
import plotly.graph_objects as go
from utils import DataLoader, Preprocessor
from models import LogisticRegression, LinearRegression, SupportVectorMachine, RandomForest, NaiveBayesian



# ------------ processing argument options
# -----------------------------------------------
parser = argparse.ArgumentParser(description='Movie Review Sentiment Analysis')
parser.add_argument('--model', action='store', type=str, help='naive_bayesian, support_vector_machine, random_forest, logistic_regression, linear_regression ', required=True)
parser.add_argument('--x-train-path', action='store', type=str, help='The training data CSV file path', required=True)
parser.add_argument('--y-train-path', action='store', type=int, help='The labeled data CSV file path', required=True)

args                = parser.parse_args()
model               = args.model
data_path           = args.x_train_path
labels_path         = args.y_train_path
preprocessed        = Preprocessor(paths=[data_path, labels_path]) 
training_dataloader = DataLoader(preprocessed=preprocessed)



# ------------ training process based on selected model
# ----------------------------------------------------------
if model == "naive_bayesian":
    nb_model = NaiveBayesian(training_dataloader=training_dataloader)
elif model == "support_vector_machine":
    svm_model = SupportVectorMachine(training_dataloader=training_dataloader)
elif model == "random_forest":
    rf_model = RandomForest(training_dataloader=training_dataloader)
elif model == "logistic_regression":
    loreg_model = LogisticRegression(training_dataloader=training_dataloader)
elif model == "linear_regression":
    lireg_mode = LinearRegression(training_dataloader=training_dataloader)
else:
    print("[?] Invalid Model!")
    sys.exit(1)