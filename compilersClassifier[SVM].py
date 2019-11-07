# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import random

import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import *
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report

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
vectorizer = CountVectorizer(ngram_range=(2, 7), binary = True)

# Slightly worst overall results
#vectorizer = TfidfVectorizer(ngram_range=(2, 7), binary = True)


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

"""# Create Model"""
#print('Fitting with GridSearch fo best model')

results = open("results.txt", "a")

"""parameters = {'kernel':['rbf'], 'C':[10]  }
modelclass = svm.SVC(gamma='scale')
gridmodel = GridSearchCV(modelclass, parameters, cv=5, iid=False, n_jobs = -1)
gridmodel.fit(X_train, y_train)

print(gridmodel.cv_results_)
for i in range(0,len(gridmodel.cv_results_['params'])):
    print("[%2d] params: %s  \tscore: %.3f +/- %.3f" %(i,
        gridmodel.cv_results_['params'][i],
        gridmodel.cv_results_['mean_test_score'][i],
        gridmodel.cv_results_['std_test_score'][i]))
    results.write('Configuration: ID:{} PARAMS:{} SCORE:{} STD_DV:{} \n'.format(i, gridmodel.cv_results_['params'][i],\
                                                                                gridmodel.cv_results_['mean_test_score'][i],\
                                                                                gridmodel.cv_results_['std_test_score'][i]))
    results.write('Kernel: {}\n'.format(gridmodel.cv_results_['params'][i]['kernel']))
    results.write('C: {}\n'.format(gridmodel.cv_results_['params'][i]['C']))

a = np.argmax(gridmodel.cv_results_['mean_test_score'])
bestparams = gridmodel.cv_results_['params'][a]
bestscore = gridmodel.cv_results_['mean_test_score'][a]

print('-------------------------------BEST MODEL------------------------------')
print("Best configuration [%d] %r  %.3f" %(a,bestparams,bestscore))
print("Best kernel: %s" %(bestparams['kernel']))
print("Best C: %s" %(bestparams['C']))
results.write('\nBest configuration: ID:{} PARAMS:{} SCORE:{}\n'.format(a, bestparams, bestscore))
results.write('Best Kernel: {}\n'.format(bestparams['kernel']))
results.write('Best C: {}\n'.format(bestparams['C']))"""

model = svm.SVC(kernel='rbf', C=10, gamma='scale', probability=True)
model.fit(X_train, y_train)

"""# Save best model"""
name = 'compilers[SVM,rbf].joblib'
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
