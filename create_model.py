# MLP for the IMDB problem
from keras.datasets import imdb
import time

# Libraries for different
from sklearn import svm
from sklearn.metrics import classification_report
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# load libraries
from keras.preprocessing import sequence

# load the dataset but only keep the top n words, zero the rest ( # load data and Set the number of words we want)
top_words = 5000
max_words = 500

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='linear', C=1)
t0 = time.time()
classifier_linear.fit(X_train, y_train)
t1 = time.time()
prediction_linear = classifier_linear.predict(X_test)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1

# results
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
report = classification_report(y_test, prediction_linear, output_dict=True)
print('positive: ', report['pos'])
print('negative: ', report['neg'])

import pickle
# Save to file in the current working directory
pkl_filename = "svm_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(classifier_linear, file)


