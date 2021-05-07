# Setup

### pip
```pip install -r requirements.txt```

### Anaconda

```conda create --name mrsa --file requirements.txt && conda activate mrsa```

And finally:

```python -m spacy download en```

# Usage

```python app.py --model support_vector_machine --train-path dataset/train --valid-path dataset/valid --test-path dataset/test```

> More options: ```python app.py --help```

### Supported Models:

* Naive Bayes Multinomial Algorithm
* Logistic Regression 
* Random Forest
* Support Vector Machine

# Preprocessing

The `MRSADatasetPipeline` class is responsible for tokenizing(lemmatization and lowercase converting), cleaning(stopwords and punctuations removal) and vectorization using BOW with TF-IDF normalization of our training, valid and testing documents.

# Training Process

Training dataloader object built on dataset object is feeded into the various ML models like **Naive Bayesian MultinomialNB**, **Logistic Regression**, **Support Vector Machine** and **Random Forest** to fill its pipeline based on preprocessed and transformed(custom `Transformer` class) data and the selected model.

On training, evaluating and predicting the data will be feeded into the dataloader pipeline to filter them through three estimators: custom cleaner transformer, `vectorizer()` method of `MRSADatasetPipeline` class which will call the `tokenizer()` method automatically and the selected model.

To train the model using the training dataloader object pipeline we called the `train()` method on the selected model in our `app.py`. After the training is over we called `self.dataloader()` object inside the `train` method in `BaseLine` class to save the vocabulary generated by TF-IDF vectorization algorithm for later usage. Below are calculated and plotted statistical results like confusion matrix for each model.

[Generated Vocabulary using TF-IDF Vectorization Algorithm](https://github.com/wildonion/mrsa/tree/main/utils/vocabulary)

# Statistical Results for Naive Bayes Multinomial

### 📊 Confusion Matrix
<p align="center">
    <img src="https://github.com/wildonion/mrsa/blob/main/utils/cmat/MultinomialNB.png">
</p>

* **Accuracy:**  0.869625

* **Precision:** 0.88301404853129

* **Recall:** 0.85526966848095

* **f1-score:** 0.8689204474048009

# Statistical Results for Logistic Regression

### 📊 Confusion Matrix
<p align="center">
    <img src="https://github.com/wildonion/mrsa/blob/main/utils/cmat/LogisticRegression.png">
</p>

* **Accuracy:**  0.885375

* **Precision:** 0.876967095851216

* **Recall:** 0.9016915910762442

* **f1-score:** 0.8891575003021878

# Statistical Results for Random Forest

### 📊 Confusion Matrix
<p align="center">
    <img src="https://github.com/wildonion/mrsa/blob/main/utils/cmat/RandomForestClassifier.png">
</p>

* **Accuracy:** 0.8565

* **Precision:** 0.8630684657671165

* **Recall:** 0.8519980266403552

* **f1-score:** 0.8574975173783516

# Statistical Results for Support Vector Machine

### 📊 Confusion Matrix
<p align="center">
    <img src="https://github.com/wildonion/mrsa/blob/main/utils/cmat/SVC.png">
</p>

* **Accuracy:** 0.8978

* **Precision:** 0.8856372737774355

* **Recall:** 0.9148766905330151

* **f1-score:** 0.9000195656427313

# Classifying Process on Test Data

For testing our models we've just passed the CSV file path of the test data through the model call pattern(`__call__`) and used the built-in piepline inside the `DataLoader` class using its `.predict()` method to predict the labels for our CSV test data. The reason for that is because the input data into the model needs to be proprocessed just like we did with the training data. Fortunately our `DataLoader` pipeline object which is filled inside of our `BaseLine` class based on preprocessed and transformed(using `Transformer` class) data and the selected model, will do the preprocessing of the feeded test data into the selected model for us automatically everytime in the first run of our app.

Then we saved our labeled data into a new CSV file based on the selected model.

[Labeled Test Data using Support Vector Machine](https://github.com/wildonion/mrsa/blob/main/utils/labeled/SVC.csv)

[Labeled Test Data using Naive Bayes Multinomial](https://github.com/wildonion/mrsa/blob/main/utils/labeled/MultinomialNB.csv)

[Labeled Test Data using Naive Random Forest](https://github.com/wildonion/mrsa/blob/main/utils/labeled/RandomForestClassifier.csv)

[Labeled Test Data using Logistic Regression](https://github.com/wildonion/mrsa/blob/main/utils/labeled/LogisticRegression.csv)

# Conclusion

We trianed all the models using their default parameters and as we can see the best accuracies are belongs to **Support Vector Machine** which has reached the state of the art of our results in classifying the sentiments for our dataset.
