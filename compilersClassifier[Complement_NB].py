# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import random

import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import *
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import *

print('Libraries imported.')


"""#Import Dataset e print it"""

filename='train_dataset.jsonl'
db = pd.read_json(filename,lines=True)
print(db)

instructionExamples_train = []

currentExample = ''

for example in db.instructions:
    for instr in example:
        mnemonics=instr.split()[0]
        currentExample += mnemonics + ' '
    instructionExamples_train.append(currentExample)
    currentExample = ''
y_all = db.compiler


print(instructionExamples_train[6])

"""#Vectorizing Dataset"""

# Better Results
#vectorizer = CountVectorizer(ngram_range=(2, 7), binary = True)

# Slightly worst overall results
vectorizer = TfidfVectorizer(ngram_range=(2, 7), binary = True)


X_all = vectorizer.fit_transform(instructionExamples_train)

"""#Split Dataset"""

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all,  test_size=0.2, random_state=15)

id = random.randrange(0,X_train.shape[0])
print('%d ' %(id))

print("Train: %d - Test: %d" %(X_train.shape[0],X_test.shape[0]))
#print('%d %s %s' %(id,str(y_train[id]),str(X_train[id])))

#FEATURE NAMES
"""names = vectorizer.get_feature_names()
print(len(names))
for i in range(1000):
    print(names[i])"""

results = open("results[COMPILERS][Comp_NB, tfidf].txt", "a")

"""# Model Cross Evaluation"""
"""cv = ShuffleSplit(n_splits=5, test_size=0.333, random_state=120)
scores = cross_val_score(model, X_all, y_all, cv=cv)
print(scores)
print("Accuracy: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))"""

"""# Create Model"""
model = ComplementNB()
model.fit(X_train, y_train)


"""# Save model"""
name = 'compilers[Comp_NB,tfidf].joblib'
joblib.dump(model, name)

#LOAD BEST MODEL
#bestModel = joblib.load(bestName)
"""# Predict on Test set"""
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
cr = classification_report(y_test, y_pred)
print(cr)

cmString = np.array2string(cm)
results.write('Title\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(cr, cm))
results.close()
