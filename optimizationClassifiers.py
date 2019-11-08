# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import random

import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit, cross_validate
from sklearn.feature_extraction.text import *
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import *
from sklearn import svm

print('Libraries imported.')

"""#Import Dataset e print it"""

filename='train_dataset.jsonl'
db = pd.read_json(filename,lines=True)
#print(db)

instructionExamples_train = []
currentExample = ''

"""# Extracts mnemonics only"""
for example in db.instructions:
    for instr in example:
        mnemonics=instr.split()[0]
        currentExample += mnemonics + ' '
    instructionExamples_train.append(currentExample)
    currentExample = ''
"""#Load labels"""
y_all = db.opt
#y_all = db.compiler

#Print a processed example
print(instructionExamples_train[random.randrange(0,len(instructionExamples_train))])

"""#Vectorizing Dataset"""
vectorizer = CountVectorizer()
n_gram_vectorizer = CountVectorizer(ngram_range=(2, 7))

print("Vectorizing")
X_all_bag_of_words = vectorizer.fit_transform(instructionExamples_train)
X_all_n_grams = n_gram_vectorizer.fit_transform(instructionExamples_train)


results = open("results[OPT].txt", "a")

"""# Create Model"""
mnb_bag_of_words = MultinomialNB()
cnb_bag_of_words = ComplementNB()
mnb_n_grams = MultinomialNB()
cnb_n_grams = ComplementNB()

"""# Models Cross Evaluation"""
print("Performing Cross evaluation")
metrics = ['accuracy','f1_macro']
chosen_score_text = 'f1_score'
chosen_score = 'test_f1_macro'
cv = ShuffleSplit(n_splits=5, test_size=0.333, random_state=342)
scores_1 = cross_validate(mnb_bag_of_words, X_all_bag_of_words, y_all, cv=cv, n_jobs = -1, scoring = metrics)
scores_2 = cross_validate(cnb_bag_of_words, X_all_bag_of_words, y_all, cv=cv, n_jobs = -1, scoring = metrics)
scores_3 = cross_validate(mnb_n_grams, X_all_n_grams, y_all, cv=cv, n_jobs = -1, scoring = metrics)
scores_4 = cross_validate(cnb_n_grams, X_all_n_grams, y_all, cv=cv, n_jobs = -1, scoring = metrics)

print("Bag of Words Representation")
print("Simple MultinomialNB Model")
#print(scores_1)
print(chosen_score_text + ": %0.3f (+/- %0.2f)" % (scores_1[chosen_score].mean(), scores_1[chosen_score].std() * 2))
print("ComplementNB Model")
#print(scores_2)
print(chosen_score_text + ": %0.3f (+/- %0.2f)" % (scores_2[chosen_score].mean(), scores_2[chosen_score].std() * 2))

print("n_grams Representation")
print("Simple MultinomialNB Model")
#print(scores_3)
print(chosen_score_text + ": %0.3f (+/- %0.2f)" % (scores_3[chosen_score].mean(), scores_3[chosen_score].std() * 2))
print("ComplementNB Model")
#print(scores_4)
print(chosen_score_text + ": %0.3f (+/- %0.2f)" % (scores_4[chosen_score].mean(), scores_4[chosen_score].std() * 2))

"""# SVM GridSearch"""
"""parameters = {'kernel':['linear', 'poly', 'rbf'], 'C':[0.01, 0.1, 1, 10, 100]  }
modelclass = svm.SVC(gamma='scale')
gridmodel = GridSearchCV(modelclass, parameters, cv=5, iid=False, n_jobs = -1)
gridmodel.fit(X_all_n_grams, y_all)

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
results.write('Best C: {}\n'.format(bestparams['C']))

svm_bag_of_words = svm.SVC(kernel=bestparams['kernel'], C=bestparams['C'], gamma='scale', probability=True)
svm_n_grams = svm.SVC(kernel=bestparams['kernel'], C=bestparams['C'], gamma='scale', probability=True)

"""
svm_bag_of_words = svm.SVC(kernel='rbf', C=10, gamma='scale', probability=True)
svm_n_grams = svm.SVC(kernel='rbf', C=10, gamma='scale', probability=True)



"""#Split Dataset for final evaluation"""
#Bag of words
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_all_bag_of_words, y_all,  test_size=0.2, random_state=15)
#n_grams
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_all_n_grams, y_all,  test_size=0.2, random_state=15)

id = random.randrange(0,X_train_1.shape[0])
print('%d ' %(id))

print("Bag of words")
print("Train: %d - Test: %d" %(X_train_1.shape[0],X_test_1.shape[0]))
#print('%d %s %s' %(id,str(y_train_1[id]),str(X_train_1[id])))

id = random.randrange(0,X_train_2.shape[0])
print('%d ' %(id))

print("n_grams")
print("Train: %d - Test: %d" %(X_train_2.shape[0],X_test_2.shape[0]))
#print('%d %s %s' %(id,str(y_train_2[id]),str(X_train_2[id])))

print("Fitting models")
mnb_bag_of_words.fit(X_train_1, y_train_1)
cnb_bag_of_words.fit(X_train_1, y_train_1)
svm_bag_of_words.fit(X_train_1, y_train_1)
mnb_n_grams.fit(X_train_2, y_train_2)
cnb_n_grams.fit(X_train_2, y_train_2)
svm_n_grams.fit(X_train_2, y_train_2)

"""# Save models"""
name = 'opt[mnb_bag_of_words].joblib'
joblib.dump(mnb_bag_of_words, name)

name = 'opt[cnb_bag_of_words].joblib'
joblib.dump(cnb_bag_of_words, name)

name = 'opt[svm_bag_of_words].joblib'
joblib.dump(svm_bag_of_words, name)

name = 'opt[mnb_n_grams].joblib'
joblib.dump(mnb_n_grams, name)

name = 'opt[cnb_n_grams].joblib'
joblib.dump(cnb_n_grams, name)

name = 'opt[svm_n_grams].joblib'
joblib.dump(svm_n_grams, name)

#LOAD BEST MODEL
#bestName = ''
#bestModel = joblib.load(bestName)

"""# Predict on Test set"""
y_pred = mnb_bag_of_words.predict(X_test_1)
cm = confusion_matrix(y_test_1, y_pred)
print(cm)
cr = classification_report(y_test_1, y_pred)
print(cr)

cmString = np.array2string(cm)
results.write('Title\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(cr, cm))

y_pred = cnb_bag_of_words.predict(X_test_1)
cm = confusion_matrix(y_test_1, y_pred)
print(cm)
cr = classification_report(y_test_1, y_pred)
print(cr)

cmString = np.array2string(cm)
results.write('Title\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(cr, cm))

y_pred = svm_bag_of_words.predict(X_test_1)
cm = confusion_matrix(y_test_1, y_pred)
print(cm)
cr = classification_report(y_test_1, y_pred)
print(cr)

cmString = np.array2string(cm)
results.write('Title\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(cr, cm))

y_pred = mnb_n_grams.predict(X_test_2)
cm = confusion_matrix(y_test_2, y_pred)
print(cm)
cr = classification_report(y_test_2, y_pred)
print(cr)

cmString = np.array2string(cm)
results.write('Title\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(cr, cm))


y_pred = cnb_n_grams.predict(X_test_2)
cm = confusion_matrix(y_test_2, y_pred)
print(cm)
cr = classification_report(y_test_2, y_pred)
print(cr)

cmString = np.array2string(cm)
results.write('Title\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(cr, cm))

y_pred = svm_n_grams.predict(X_test_2)
cm = confusion_matrix(y_test_2, y_pred)
print(cm)
cr = classification_report(y_test_2, y_pred)
print(cr)

cmString = np.array2string(cm)
results.write('Title\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(cr, cm))

results.close()
