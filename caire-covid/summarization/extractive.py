from extractive import Summarizer
import argparse
import sys
import pandas as pd
import csv
import requests
import numpy as np
import scipy
import json

def get_ir_result(query):
    post = {}
    post['text'] = query
    r = requests.post('http://hlt027.ece.ust.hk:5000/query_paragraph', json=post)
    paragraphs = []
    for i in range(len(r.json())):
        if 'paragraphs' in r.json()[i].keys():
            paragraphs.append(r.json()[i]['paragraphs'][0]['text'])
    return paragraphs

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='unilm summary')
    parser.add_argument('-q','--query', help='input query', required=False, default='What is the range of incubation periods for COVID-19 in humans?')
    args = vars(parser.parse_args())

    # return all paragraphs
    query = args['query']
    paragraphs = get_ir_result(query)

    # get all article
    article = ""
    for paragraph in paragraphs:
        article += paragraph

    # initial model
    model = Summarizer(
        model='albert-large-v1',
        hidden= -2,
        reduce_option= 'mean'
    )

    # get  article information from cluster
    sentences, n_clusters, labels, hidden_args, content, hidden = model(article)
    # get query information from cluster
    _,_,_,_,_,query_hidden = model(query)

    # get candidates representation
    candidates_hidden = np.zeros((n_clusters, hidden.shape[1]))
    for i in range(n_clusters):
        candidates_hidden[i] = hidden[hidden_args[i]]
    distances = scipy.spatial.distance.cdist(query_hidden, candidates_hidden, "cosine")[0].tolist()

    # rerank  the center of each article by distance with query
    rank = np.argsort(distances)

    result_cluster = []
    for i in range(len(rank)):
        idx = int(np.argwhere(rank == i))
        result_cluster.append(content[hidden_args[idx]])
        # print(content[hidden_args[idx]])

    # rerank all sentence only by distance with query
    result_query = []
    distances = scipy.spatial.distance.cdist(query_hidden, hidden, "cosine")[0].tolist()
    rank = np.argsort(distances)
    for i in range(len(rank)):
        idx = int(np.argwhere(rank == i))
        result_query.append(content[idx])
        # print(content[idx])

    with open('./save/{}.json'.format(query), 'w') as f:
        for i in range(len(result_cluster)):
            one_line = {}
            one_line["number"] = i
            one_line["answer"] = result_cluster[i]
            json.dump(one_line, f)
            f.write('\n')
        for i in range(len(result_query)):
            one_line = {}
            one_line["number"] = i
            one_line["answer"] = result_query[i]
            json.dump(one_line, f)
            f.write('\n')