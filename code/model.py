# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 08:06:17 2019

@author: sarat
"""

from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize



from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from nltk.stem import PorterStemmer
ps = PorterStemmer()


import numpy as np
import query as q

#%%
def preprocess(doc):
    words = word_tokenize(doc)
    mod_words = []
    for word in words:
        word = word.lower()
        #word = lemmatizer.lemmatize(word)
        
        word = ps.stem(word) 
        symbols = ['.',',','?','!']
        if word[-1] in symbols:
            word = word[:-1]
        
        if len(word) > 1 and word not in stop_words:
                mod_words.append(word)
    return mod_words


def calculate_similarity(data):
    scores = []
    
    for i in range(len(data)):
        query_score = []
        ques = data.iloc[i]['question']
        for j in range(1,5):
            opt = data.iloc[i]['choice'+str(j)]
            query = ques + " " + opt
            query = preprocess(query)
            _,score = model.docvecs.most_similar([model.infer_vector(query)])[0] # max similar one
            
            query_score.append(score)
        scores.append(query_score)
    return scores

def find_max(score):
    max_val = max(score)
    ind = []
    for i in range(len(score)):
        c = ''
        if score[i] == max_val:
            if i == 0:
                c = 'A'
            elif i == 1:
                c = 'B'
            elif i == 2:
                c = 'C'
            else:
                c = 'D'
            ind.append(c)
    return ind


def find_labels(scores):
    labels = []
    for score in scores:
        label = find_max(score)
        labels.append(label)
    return labels


def find_accuracy(labels,answers):
    count = 0
    n = len(answers)
    for i in range(n):
        if answers[i] in labels[i]:
            l = len(labels[i])
            count += 1.0 / l
    acc = count / n
    return acc

#%%

model= Doc2Vec.load("./model/d2v.model")

path = 'Challenge.jsonl'
query_data = q.load_json(path)
scores = calculate_similarity(query_data)

labels = find_labels(scores)
acc = find_accuracy(labels,query_data['answer'])
print('Accuracy using Doc2Vec model :',acc)