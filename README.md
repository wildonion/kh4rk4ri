# Setup

### pip
```pip install -r requirements.txt```

### Anaconda

```conda create --name mrsa --file requirements.txt && conda activate mrsa```

And finally:

```python -m spacy download en```

# Usage

```console 
python app.py --model support_vector_machine --train-path dataset/train --valid-path dataset/valid --test-path dataset/test
```

> More options: ```python app.py --help```

### Supported Models:

* Naive Bayes Multinomial
* Logistic Regression 
* Random Forest
* Support Vector Machine
* SGD Classifier

# Abstract

One of the most challenging analytical problems in the world of natural language processing is the analysis of hidden emotions in the text such as news or user comments about a specific subject like movies of a website.
Therefor due to the high volume of data collected by web servers around the world, using mathematical model tools based on AI technology can be a great solution to solve this problem.
In this project we have tried to train various mathematical models in order to reach the state of the art of the result accuracy on our review datasets.
In the following, we will review the steps performed to train these models and examine each of these models


# Preprocessing

The `MRSADatasetPipeline` class is responsible for tokenizing(lemmatization and lowercase converting), cleaning(stopwords and punctuations removal) and vectorization using BOW with TF-IDF normalization of our training, valid and testing documents.

# Training Process

Training dataloader object which is built on dataset object is feeded into the various ML models like **Naive Bayesian MultinomialNB**, **Logistic Regression**, **Support Vector Machine**, **SGD Classifier** and **Random Forest** to fill its pipeline based on preprocessed and transformed(custom `Transformer` class) data and the selected model.

On training, evaluating and predicting; the data will be feeded into the dataloader pipeline, in order to filter them through three estimators: custom cleaner transformer, `vectorizer()` method of `MRSADatasetPipeline` class which will call the `tokenizer()` method automatically and the selected model.

To train the model using the training dataloader object pipeline we called the `train()` method on the selected model in our `app.py`. After the training is over we called `self.__save()` method inside the `train` method in `BaseLine` class to save the whole `BaseLine` class for later tests. Below are calculated and plotted statistical results like confusion matrix for each model.

[BaseLine Class](https://github.com/wildonion/mrsa/tree/main/utils/model)

# Statistical Results for Naive Bayes Multinomial

### 📊 Confusion Matrix
<p align="center">
    <img src="https://github.com/wildonion/mrsa/blob/main/utils/cmat/MultinomialNB.png">
</p>

* **Accuracy:**  0.8662

* **Precision:** 0.8751525010166734

* **Recall:** 0.8560063643595863

* **f1-score:** 0.8654735572089283

* **ROC AUC score:** 0.866257405832247

# Statistical Results for Logistic Regression

### 📊 Confusion Matrix
<p align="center">
    <img src="https://github.com/wildonion/mrsa/blob/main/utils/cmat/LogisticRegression.png">
</p>

* **Accuracy:**  0.8922

* **Precision:** 0.8799538283955367

* **Recall:** 0.9097056483691328

* **f1-score:** 0.8945824369254841

* **ROC AUC score:** 0.8921014163004152

# Statistical Results for Random Forest

### 📊 Confusion Matrix
<p align="center">
    <img src="https://github.com/wildonion/mrsa/blob/main/utils/cmat/RandomForestClassifier.png">
</p>

* **Accuracy:** 0.8534

* **Precision:** 0.8543573418225229

* **Recall:** 0.8540175019888624

* **f1-score:** 0.8541873881042371

* **ROC AUC score:** 0.853396522514946

# Statistical Results for SGD Classifier

### 📊 Confusion Matrix
<p align="center">
    <img src="https://github.com/wildonion/mrsa/blob/main/utils/cmat/SGDClassifier.png">
</p>

* **Accuracy:** 0.8948

* **Precision:** 0.8864696734059098

* **Recall:** 0.9069212410501193

* **f1-score:** 0.8965788438851748

* **ROC AUC score:** 0.8947317387873284

# Statistical Results for Support Vector Machine

### 📊 Confusion Matrix
<p align="center">
    <img src="https://github.com/wildonion/mrsa/blob/main/utils/cmat/SVC.png">
</p>

* **Accuracy:** 0.8986

* **Precision:** 0.888201160541586

* **Recall:** 0.9132856006364359

* **f1-score:** 0.9005687389684252

* **ROC AUC score:** 0.8985172975024497

# Classifying Process on Test Data

For testing our models we've just passed the CSV file path of the test data through the model call pattern(`__call__`) and used the built-in piepline inside the `DataLoader` class using its `.predict()` method to predict the labels for our CSV test data. The reason for that is because the input data into the model needs to be proprocessed just like we did with the training data. Fortunately our `DataLoader` pipeline object which is filled inside of our `BaseLine` class based on preprocessed and transformed(using `Transformer` class) data and the selected model, will do the preprocessing of the feeded test data into the selected model for us automatically everytime in the first run of our app.

 > Labeled test data into a new CSV file based on the selected model:

[Labeled Test Data using Support Vector Machine](https://github.com/wildonion/mrsa/blob/main/utils/labeled/SVC.csv)

[Labeled Test Data using Naive Bayes Multinomial](https://github.com/wildonion/mrsa/blob/main/utils/labeled/MultinomialNB.csv)

[Labeled Test Data using Naive Random Forest](https://github.com/wildonion/mrsa/blob/main/utils/labeled/RandomForestClassifier.csv)

[Labeled Test Data using Logistic Regression](https://github.com/wildonion/mrsa/blob/main/utils/labeled/LogisticRegression.csv)

[Labeled Test Data using SGD Classifier](https://github.com/wildonion/mrsa/blob/main/utils/labeled/SGDClassifier.csv)

# Conclusion

As we can see the best accuracies are belongs to **Support Vector Machine** with **kernel** and **C** equal to `rbf` and `1E5` respectively which has reached the state of the art of our results in classifying the sentiments for our dataset.
