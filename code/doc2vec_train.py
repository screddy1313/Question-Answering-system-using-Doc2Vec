# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 07:53:22 2019

@author: sarat
"""

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pickle

import os



def load_pickle_data(path):
    path = os.path.join('pickle',path)
    out = open(path,'rb')
    data = pickle.load(out)
    return data

docs = load_pickle_data('docs.pkl')
vocab = load_pickle_data('vocab.pkl')

#%%

tag_data = [TaggedDocument(words= d, tags=[str(i)]) for i, d in enumerate(docs)]

max_epochs = 50
vector_size = len(vocab)
alpha = 0.02

model = Doc2Vec(size=vector_size,
                alpha=alpha, 
                min_alpha=0.0001,
                min_count=1,
                dm =1)
  
model.build_vocab(tag_data)

for epoch in range(max_epochs):
    
    model.train(tag_data,
                total_examples=model.corpus_count,
                epochs=model.iter)

    model.alpha -= 0.0003
    model.min_alpha = model.alpha
    print('iteration {0}'.format(epoch))
    
model.save("./model/d2v.model")

print("Model is Saved")