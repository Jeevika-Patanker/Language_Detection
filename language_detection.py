from flask import Flask,request,render_template,jsonify
import os
import joblib
import pandas as pd

language_codes = {
    'ar': 'Arabic',
    'bg': 'Bulgarian',
    'de': 'German',
    'el': 'Modern Greek',
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'hi': 'Hindi',
    'it': 'Italian',
    'ja': 'Japanese',
    'nl': 'Dutch',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'sw': 'Swahili',
    'th': 'Thai',
    'tr': 'Turkish',
    'ur': 'Urdu',
    'vi': 'Vietnamese',
    'zh': 'Chinese'
}






app = Flask(__name__)


def get_language_full_form(code):
    return language_codes.get(code, 'Unknown')


def predict_classes(text):
    model_path= os.getcwd()+r'\model'
    vector,classifer = joblib.load(model_path+r'\classifier.pkl')
    
    text=vector.transform(text)
    prediction= classifer.predict(text)

    return prediction[0]


@app.route('/')
def index():
    return render_template('index.html')        

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        result = request.form
        print(result)
        content =request.form['text']
        print(content)
        print(content)
        text = pd.Series(content)
        print(text)
        prediction=predict_classes(text)
        prediction=get_language_full_form(prediction)
    return render_template('index.html',pred=prediction)

if __name__ == '__main__':
    app.run(debug=True,port=8080)