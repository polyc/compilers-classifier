
import numpy as np
import pandas as pd
import random

import joblib

from sklearn.feature_extraction.text import *

print('Libraries imported.')

"""#Import Dataset e print it"""
filename = 'train_dataset.jsonl'
db = pd.read_json(filename,lines=True)
#print(db)

filename='test_dataset_blind.jsonl'
test_db = pd.read_json(filename,lines=True)

instructionExamples_train = []
currentExample = ''

"""# Extracts mnemonics only"""
for example in db.instructions:
    for instr in example:
        mnemonics=instr.split()[0]
        currentExample += mnemonics + ' '
    instructionExamples_train.append(currentExample)
    currentExample = ''

instructionExamples_test = []
currentExample = ''

"""# Extracts mnemonics only"""
for example in db.instructions:
    for instr in example:
        mnemonics=instr.split()[0]
        currentExample += mnemonics + ' '
    instructionExamples_test.append(currentExample)
    currentExample = ''

#Print a processed examples
print(instructionExamples_train[random.randrange(0,len(instructionExamples_train))])
print(instructionExamples_test[random.randrange(0,len(instructionExamples_test))])

n_gram_vectorizer = CountVectorizer(ngram_range=(2, 7))
n_gram_vectorizer.fit(instructionExamples_train)
X_test_n_grams = n_gram_vectorizer.transform(instructionExamples_test)

#LOAD BEST MODEL
bestName = 'opt[cnb_n_grams].joblib'
bestModel = joblib.load(bestName)

results_name = "blind_set_results[opt].csv"
results = open(results_name, "a")

"""# Predict on Test set"""
y_pred = bestModel.predict(X_test_n_grams)
df = pd.DataFrame(y_pred)
df.to_csv(path_or_buf = results, index=False)
results.close()
