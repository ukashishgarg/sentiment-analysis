# SAVE THE MODEL TO PICKLE FILE
import pickle
from keras.preprocessing import sequence
from sklearn.feature_extraction.text import CountVectorizer

# LOAD SAVED PICKLE MODEL AND TEST IT WITH SOME RANDOM TEXT
text = 'Not good'
max_words = 500
text = [text]

cv = CountVectorizer()
text_vect = cv.fit_transform(text).toarray()
text_vect = sequence.pad_sequences(text_vect, maxlen=max_words)

mdl = pickle.load(open('rf_model.pkl','rb'))
prediction = mdl.predict(text_vect)

if (prediction == 0):
    print('Positive')
else:
    print('Negative')