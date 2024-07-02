# Language_Detection
Language Identification Project

Overview:-


This project aims to develop a language identification model using the Papluca language identification dataset. The model uses a pipeline approach, combining TF-IDF vectorization with a Multinomial Naive Bayes classifier to predict the language of a given text.

Dataset:-


The dataset used in this project is the Papluca language identification dataset, which contains text samples in various languages. The dataset is split into training and testing sets.

Model:-

The model consists of the following components:

Text Preprocessing:-The text data is cleaned by removing special characters, numbers, and converting to lowercase.

TF-IDF Vectorization:- The cleaned text data is vectorized using TF-IDF vectorization.

Multinomial Naive Bayes Classifier:- The vectorized data is fed into a Multinomial Naive Bayes classifier to predict the language.


Evaluation:-

The model is evaluated using accuracy, classification report, and confusion matrix.


Usage:-

To use the model, simply run the provided Python script. The script will train the model on the training data and evaluate its performance on the testing data. You can also use the model to make predictions on custom input text.


Example:-


To make predictions on custom input text, use the following code:


example_text = np.array(["जापान"])
new_predictions = language_detector.predict(example_text)
print("Predictions for custom input:", new_predictions)


Results:-


The model achieves an accuracy of [insert accuracy score] on the testing data.


Future Work:-


Improve the model's performance by experimenting with different preprocessing techniques, vectorization methods, and classification algorithms.
Expand the dataset to include more languages and text samples.


Dependencies:-


Python 3.x
scikit-learn
pandas
numpy
matplotlib
seaborn
datasets
