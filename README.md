# Setup

### pip
```pip install -r requirements.txt```

### Anaconda

```conda create --name mrsa --file requirements.txt && conda activate mrsa```

And finally:

```python -m spacy download en```

# Usage

```python app.py --model naive_bayesian --train-path dataset/train --test-path dataset/test```

> More options: ```python app.py --help```

# Dataset

[Movie Review Dataset](https://github.com/HamedBabaei/ML992/tree/main/dataset)

# Preprocessing

The `MRSADatasetPipeline` class is responsible for tokenizing(lemmatization and lowercase converting), cleaning(stopwords and punctuations removal) and vectorization using BOW with TF-IDF normalization.

Preprocessed input texts of our training documents are passed through the `DataLoader` object to split the training documents into `x_train`, `y_train`, `x_test` and `y_test` with 20 percent of test data.

# Training Process

After that the dataloader object is feeded into the various ML models like **Naive Bayesian GaussianNB and MultinomialNB**, **Logistic and Linear Regression**, **Support Vector Machine** and **Random Forest** to fill its pipeline based on preprocessed and transformed(using `Transformer` class) data and the selected model.

Finally we called the `train()` method on the selected model to train it using the training dataloader object pipeline. Below are calculated and plotted statistical results like confusion matrix for each model. 

# Statistical Results

### 📊 Confusion Matrix for Naive Bayes Multinomial Algorithm
<p align="center">
    <img src="https://github.com/wildonion/mrsa/blob/main/utils/cmat/MultinomialNB.png">
</p>

* **Accuracy** : 0.869625

* **Precision** : 0.88301404853129

* **Recall** : 0.85526966848095

* **f1_score** : 0.8689204474048009

[Confusion Matrix for other Models](https://github.com/wildonion/mrsa/tree/main/utils/cmat)

# Classifying Process on Test Data

For testing our models we've just passed the CSV file path of the test data through the model call pattern(`__call__`) and used the built-in piepline inside the `DataLoader` class using its `.predict()` method to predict the labels for our CSV test data. The reason for that is because the input data into the model needs to be proprocessed just like we did with the training data. Fortunately our `DataLoader` pipeline object which is filled inside of our `BaseLine` class based on preprocessed and transformed(using `Transformer` class) data and the selected model, will do the preprocessing of the feeded test data into the selected model for us automatically everytime in the first run of our app.

Then we saved our labeled data into a new CSV file based on the selected model.

[Labeled Test Data](https://github.com/wildonion/mrsa/tree/main/utils/labeled)
