# Importing the Libraries
import numpy as np
from flask import Flask, request, render_template
from flask_cors import CORS
import os
from sklearn.externals import joblib
import pickle
import flask
import os
import newspaper
from newspaper import Article
import urllib

# Loading Flask and assigning the model variable
app = Flask(__name__)
CORS(app)
app = flask.Flask(__name__, template_folder='templates')

with open('pickle_model_senti.pkl', 'rb') as handle:
    model = pickle.load(handle)

@app.route('/')
def main():
    return render_template('main.html')


# Receiving the input url from the user and using Web Scrapping to extract the news content
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    url = request.get_data(as_text=True)[5:]
    text = urllib.parse.unquote(url)

    # LOAD SAVED PICKLE MODEL AND TEST IT WITH SOME RANDOM TEXT
    from sklearn.feature_extraction.text import CountVectorizer
    from keras.preprocessing import sequence

    # mdl = pickle.load(open('pickle_model_senti.pkl', 'rb'))
    cv = CountVectorizer()
    text = [text]
    text_vect = cv.fit_transform(text).toarray()

    max_words = 500
    text_vect = sequence.pad_sequences(text_vect, maxlen=max_words)
    pred = model.predict(text_vect)
    outcome = "Positive"
    if pred == 0:
        outcome = "Positive"
    else:
        outcome = "Negative"

    # Passing the news article to the model and returing whether it is Fake or Real
    # pred = model.predict([news])
    # return render_template('main.html', prediction_text='The news is "{}"'.format(outcome))
    return render_template('main.html', prediction_text='It is the '+outcome+' sentiment')


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)