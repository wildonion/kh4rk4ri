# Setup

```pip install -r requirements.txt```

# Usage

```python app.py --model naive_bayesian_gauss --train-path dataset/train --test-path dataset/test```

> More options: ```python app.py --help```

# Dataset

> [Project Explanation](https://github.com/HamedBabaei/ML992/tree/main/dataset)

# Preprocessing

The `Preprocessor` class is responsible for tokenizing(lemmatization and lowercase converting), cleaning(stopwords and punctuations removal), vectorizing(BOW and TF-IDF) and splitting the training documents into `x_train`, `y_train`, `x_test` and `y_test` with 20 percent of test data.

# Training Process

Preprocessed input texts of our training documents are passed through the `DataLoader` object to build the pipeline using `sklearn.pipeline` then feeded into the various ML models like **Naive Bayesian GaussianNB and MultinomialNB**, **Logistic and Linear Regression**, **Support Vector Machine** and **Random Forest**.

We trained all mentioned ML models on training dataloader object pipeline. Below are calculated and plotted statistical results for each model. 

> Generally we first build the dataset pipeline then we pass it through the dataloader object. Here we did the opposite.

# Statistical Results

# Classifying Process on Test Data