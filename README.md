# Setup

### pip
```pip install -r requirements.txt```

### Anaconda

```conda create --name mrsa --file requirements.txt && conda activate mrsa```

And finally:

```python -m spacy download en```

# Usage

```python app.py --model naive_bayesian_gauss --train-path dataset/train --test-path dataset/test```

> More options: ```python app.py --help```

# Dataset

[Movie Review Dataset](https://github.com/HamedBabaei/ML992/tree/main/dataset)

# Preprocessing

The `MRSADatasetPipeline` class is responsible for tokenizing(lemmatization and lowercase converting), cleaning(stopwords and punctuations removal) and vectorization using BOW with TF-IDF normalization.

Preprocessed input texts of our training documents are passed through the `DataLoader` object to split the training documents into `x_train`, `y_train`, `x_test` and `y_test` with 20 percent of test data.

# Training Process

After that the dataloader object is feeded into the various ML models like **Naive Bayesian GaussianNB and MultinomialNB**, **Logistic and Linear Regression**, **Support Vector Machine** and **Random Forest** to fill its pipeline based on preprocessed and transformed(using `Transformer` and `ToDense` classes) data and the selected model.

Finally we called the `train()` method on the selected model to train it using the training dataloader object pipeline. Below are calculated and plotted statistical results for each model. 

# Statistical Results

# Classifying Process on Test Data
