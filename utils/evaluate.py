# -*- coding: utf-8 -*-

from numpy import log2
from random import sample
import numpy as np
import math


def get_dcg(ordered_labels):
    return np.sum((2 ** ordered_labels - 1) / np.log2(np.arange(ordered_labels.shape[0]) + 2))


def get_idcg(complete_labels, max_len):
    return get_dcg(np.sort(complete_labels)[:-1 - max_len:-1])


def get_single_ndcg_for_rankers(descending_rankings, document_labels, max_len, idcg=None):
    if idcg == None:
        idcg = get_idcg(document_labels, max_len)
    if idcg == 0:
        return np.zeros(descending_rankings.shape[0])
    return get_single_dcg_for_rankers(descending_rankings, document_labels, max_len)/idcg


def get_single_dcg_for_rankers(descending_rankings, document_labels, max_len):
    displayed_rankings = descending_rankings[:, :max_len]
    displayed_labels = document_labels[displayed_rankings]
    return np.sum((2 ** displayed_labels - 1) / np.log2(np.arange(displayed_labels.shape[1])
                  + 2)[None, :], axis=1)


# Calculate NDCG with a single array of descending ranking and the corresponding labels.
def get_ndcg_with_labels(ranking, labels, max_len, initial, is_test = False, impression = -1):

    # if is_test and initial == 280:
    #     print labels

    idcg = get_idcg(np.asarray(labels), max_len)
    if idcg==0:
        return 0
    nominators = [2. ** label - 1. for label in labels]
    denominators = [math.log(r+2., 2) for r in ranking]
    for i in range(len(ranking)):
        if ranking[i]>=max_len:
            nominators[i] = 0

    ndcg = 0
    for i in range(len(nominators)):
        ndcg += (nominators[i] / denominators[i]) / idcg

    if impression % 100 == 0:
        print ndcg
    return ndcg

# Calculate NDCG with a single array of descending model_ranking and a descending ideal_ranking.
# Assume that the first "max_len" documents in ideal_ranking has relevance score 1 and other documents
# have relevance score 0.
def get_ndcg_with_ranking(model_ranking, ideal_ranking, max_len, initial):
    labels = [0 for i in range(len(model_ranking))]
    displayed_ideal_ranking = ideal_ranking[:max_len]
    for document in model_ranking:
      if document in displayed_ideal_ranking and document<len(model_ranking):
          labels[document] = 1

    return get_ndcg_with_labels(model_ranking, labels, max_len, initial)

def get_ndcg_with_ranking2(model_ranking, ideal_ranking, max_len, initial, impression):

    r = [0 for i in range(len(model_ranking))]      #  [2,3,4,1,0]  => [5, 4, 0, 1, 3]
    # print model_ranking
    for i in range(len(model_ranking)):
        if model_ranking[i] < len(r):
            r[model_ranking[i]] = i

    labels = [0 for i in range(len(model_ranking))]
    displayed_ideal_ranking = []

    displayed_ideal_ranking = ideal_ranking[:5]
    
    # if impression%2 == 0:
    #     displayed_ideal_ranking = ideal_ranking[:5]
    # else:
    #     displayed_ideal_ranking = ideal_ranking[:max_len]

    for document in r:
      if document in displayed_ideal_ranking and document<len(model_ranking):
          labels[document] = 1

    # if(initial == 280):
    #     print "Len of displayed: ", len(displayed_ideal_ranking)
    #     print "Attack: ", labels

    ndcg = get_ndcg_with_labels(model_ranking, labels, max_len, initial, impression)

    # if impression == 100 and initial%10 == 0:
    #     print "Model ranking: ", r
    #     print "Ideal_ranking/attacker: ", ideal_ranking
    #     print "lables: ", labels
    #     print "NDCG: ", ndcg
    #     print "------------------------------------------------"
    
    return ndcg


def evaluate_ranking(ranking, labels, idcg, max_len):
    ordered_labels = labels[ranking]
    if idcg == 0.0:
        return 0.0
    return get_dcg(ordered_labels) / idcg


def evaluate(rankings, label_vector, idcg_vector, n_queries, max_len):
    '''
    Takes rankings as lists of indices, which corresponds to label_lists, lists of label lists.
    '''
    nominators = 2. ** label_vector - 1.
    denominators = np.log2(rankings + 2.)
    nominators[idcg_vector == 0] = 0
    nominators[rankings >= max_len] = 0

    idcg_copy = np.copy(idcg_vector)
    idcg_copy[idcg_vector == 0] = 1

    return np.sum(nominators / denominators / idcg_copy) / n_queries


def get_dcg_from_matrix(label_matrix, n_vector, max_len):
    label_matrix = label_matrix[:, :max_len]

    nominators = 2 ** label_matrix - 1
    nominators[np.arange(max_len)[None, :] >= n_vector[:, None]] = 0

    denominator = np.log2(np.arange(max_len) + 2)
    idcg_vector = np.sum(nominators / denominator[None, :], axis=1)

    return idcg_vector


def get_idcg_list(label_vector, qptr, max_len, spread=False):

    n = qptr[1:] - qptr[:-1]
    max_documents = np.max(n)

    starts = np.zeros(n.shape[0] + 1, dtype=np.int32)
    starts[1:] = np.cumsum(n)

    ind = starts[:-1, None] + np.arange(0, max_documents)[None, :]
    ind = np.minimum(ind, starts[1:, None] - 1)

    label_matrix = label_vector[ind]
    label_matrix[np.arange(max_documents)[None, :] >= n[:, None]] = 0
    label_matrix = np.sort(label_matrix, axis=1)[:, ::-1]

    idcg_list = get_dcg_from_matrix(label_matrix, n, max_len)

    if spread:
        spread_ind = np.zeros(qptr[-1], dtype=np.int32)
        spread_ind[qptr[1:-1]] = 1
        spread_ind = np.cumsum(spread_ind)

        return idcg_list[spread_ind]
    else:
        return idcg_list
