# -*- coding: utf-8 -*-
# !/usr/bin/env python
#export PYTHONIOENCODING=utf-8
import datetime
import re
import random
from elasticsearch import Elasticsearch
import torch
from torch.autograd import Variable
import numpy as np
import torch.functional as F
import torch.nn.functional as F
from datasketch import MinHashLSH
from datasketch import MinHash
import pprint


def connect_elasticsearch():
	_es = None
	_es = Elasticsearch([{'host':'localhost','port':9200}])
	if _es.ping():
		print("CONECTED YAY")
	else:
		print("NOT CONECTED EHHH")
	return _es

def text_normalize(text):
	text = text.lower()
	reg = re.compile('[^а-яё ]')
	text = reg.sub('', text)
	text = re.sub(r'\s+', ' ', text )
	return text


def search(es_object, index_name, search):
	res = es_object.search(index=index_name, body=search)
	return res["hits"]["hits"]


def get_input_layer(word_idx):
    x = torch.zeros(vocabulary_size).float()
    x[word_idx] = 1.0
    return x

def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters

def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu

def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))

def find_centers(X, K):
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        clusters = cluster_points(X, mu)
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, clusters)


es = connect_elasticsearch()
search_object = {'query': {'match_all': {}}}
result = search(es, 'news', json.dumps(search_object))

for i in range(len(result)):
	res = text_normalize(result[i]["_source"]["full_text"])
	f = open("data/text" + str(i), "w")
	f.write(res)
	f.close()
	tokens = res.split()
	vocabulary = []
	for token in tokens:
		if token not in vocabulary:
			vocabulary.append(token)
	word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
	idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}
	vocabulary_size = len(vocabulary)
	window_size = 2
	idx_pairs = []
	indices = [word2idx[token] for token in tokens]
	for center_word_pos in range(len(indices)):
		for w in range(-window_size, window_size + 1):
			context_word_pos = center_word_pos + w
			if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
				continue
			context_word_idx = indices[context_word_pos]
			idx_pairs.append((indices[center_word_pos], context_word_idx))
	idx_pairs = np.array(idx_pairs)
	(mu, clusters) = find_centers(list(idx_pairs), 50) #создание кластеров
	embedding_dims = 5
	W1 = Variable(torch.randn(embedding_dims, vocabulary_size).float(), requires_grad=True)
	W2 = Variable(torch.randn(vocabulary_size, embedding_dims).float(), requires_grad=True)
	num_epochs = 100
	learning_rate = 0.001
	for epo in range(num_epochs):
		loss_val = 0
		for data, target in idx_pairs:
			x = Variable(get_input_layer(data)).float()
			y_true = Variable(torch.from_numpy(np.array([target])).long())
			z1 = torch.matmul(W1, x)
			z2 = torch.matmul(W2, z1)
			log_softmax = F.log_softmax(z2, dim=0)
			loss = F.nll_loss(log_softmax.view(1,-1), y_true)
			loss_val += loss.item()
			loss.backward()
			W1.data -= learning_rate * W1.grad.data
			W2.data -= learning_rate * W2.grad.data
			W1.grad.data.zero_()
			W2.grad.data.zero_()
		if epo % 10 == 0:
			if (loss_val/len(idx_pairs) < 3.5):
				print("Studied success")
				break
set1 = set(['то', 'и', 'как', 'может', 'получить', 'выплаты', 'спецвыпуск', 'журнала'])
set2 = set(['то', 'и', 'как', 'может', 'получить', 'выплаты', 'спецвыпуск', 'журнала'])
set3 = set(['может', 'то', 'и', 'как', 'получить', 'выпалт', 'спецвыпуска', 'журналы'])

m1 = MinHash(num_perm=128)
m2 = MinHash(num_perm=128)
m3 = MinHash(num_perm=128)
for d in set1:
    m1.update(d.encode('utf8'))
for d in set2:
    m2.update(d.encode('utf8'))
for d in set3:
    m3.update(d.encode('utf8'))
lsh = MinHashLSH(threshold=0.5, num_perm=128)
lsh.insert("m2", m2)
lsh.insert("m3", m3)
result = lsh.query(m1)
print("Approximate neighbours with Jaccard similarity > 0.5", result)
			

