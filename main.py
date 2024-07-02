import os
from dataloader.dataload import load_dataset
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score
from sklearn.metrics import precision_score,roc_auc_score,f1_score,recall_score
import pandas as pd
import joblib






#loading dataset
path=os.getcwd()+r'\data'
data=  load_dataset(path + r'\language_detection_without_Japanese.csv') 

print(type(data))

#split data
x_train,x_test,y_train,y_test = train_test_split(data['text'],data['labels'],test_size=0.3,random_state=42,shuffle=True)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(x_train)
X_test_tfidf = vectorizer.transform(x_test)


# Naive Bayes Model
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)
# Predictions
y_pred = clf.predict(X_test_tfidf)


#evaluation of the base model
print("accuracy score on test data for multinomialNB  = %s"%  accuracy_score(y_test, y_pred))


model_path=os.getcwd()+r'\model'
# joblib.dump(cleaner,model_path+r'\data_cleaner.pkl',compress=True)
# joblib.dump(Tfidf_Vector,model_path+r'\tfidf_vector.pkl',compress=True)
joblib.dump((vectorizer, clf),model_path+r'\classifier.pkl',compress=True)
print('model successfully extracted')