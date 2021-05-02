

import argparse
import plotly.graph_objects as go
from utils import DataLoader, Preprocessor, Transformer
from models import LogisticRegression, LinearRegression, SupportVectorMachine, RandomForest, NaiveBayesian


# ------------ processing argument options
# -----------------------------------------------
parser = argparse.ArgumentParser(description='Movie Review Sentiment Analysis')
parser.add_argument('--model', action='store', type=str, help='naive_bayesian_gauss, naive_bayesian_multinomial, support_vector_machine, random_forest, logistic_regression, linear_regression ', required=True)
parser.add_argument('--train-path', action='store', type=str, help='The training data CSV file path', required=True)
parser.add_argument('--test-path', action='store', type=str, help='The data CSV file path to test the pre-trained model', required=True)

args                = parser.parse_args()
model               = args.model
train_path          = args.train_path
test_path           = args.test_path
preprocessed        = Preprocessor(paths=[train_path])
print(preprocessed()) # NOTE - preprocessed() will return the self
training_dataloader = DataLoader(preprocessed=preprocessed, transformer=Transformer)



# ------------ training process based on selected model
# ----------------------------------------------------------
if model == "naive_bayesian_gauss":
    model = NaiveBayesian(training_dataloader=training_dataloader, algorithm="gauss")
if model == "naive_bayesian_multinomial":
    model = NaiveBayesian(training_dataloader=training_dataloader, algorithm="multimonial")
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
model.stat()