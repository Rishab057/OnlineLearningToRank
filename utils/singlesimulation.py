# -*- coding: utf-8 -*-

import time
import numpy as np
from evaluate import *
# from evaluate import get_idcg_list, evaluate, evaluate_ranking, get_ndcg_with_ranking
from clicks import *
import scipy.stats as stats # For kendall tau
import matplotlib.pyplot as plt
import math
import utils.rankings as rnk


class SingleSimulation(object):

  def __init__(self, sim_args, output_queue, click_model, datafold):
    self.train_only = sim_args.train_only
    self.n_impressions = sim_args.n_impressions

    self.n_results = sim_args.n_results
    self.click_model = click_model
    self.datafold = datafold
    if not self.train_only:
      self.test_idcg_vector = get_idcg_list(self.datafold.test_label_vector,
                                            self.datafold.test_doclist_ranges,
                                            self.n_results, spread=True)
    self.train_idcg_vector = get_idcg_list(self.datafold.train_label_vector,
                                           self.datafold.train_doclist_ranges,
                                           self.n_results)

    self.run_details = {
            'data folder': str(self.datafold.data_path),
            'held-out data': str(self.datafold.heldout_tag),
            'click model': self.click_model.get_name(),
          }
    self.output_queue = output_queue

    self.print_frequency = sim_args.print_freq
    self.print_all_train = sim_args.all_train
    self.print_logscale = sim_args.print_logscale
    if self.print_logscale:
      self.print_scale = self.print_frequency
      self.print_next_scale = self.print_scale
      self.print_frequency = 1

    self.last_print = 0
    self.next_print = 0
    self.online_score = 0.0
    self.cur_online_discount = 1.0
    self.online_discount = 0.9995

  def timestep_evaluate(self, results, iteration, ranker, ranking_i,
                        train_ranking, ranking_labels):

    test_print = (not self.train_only
                  and (iteration == self.last_print
                       or iteration == self.next_print
                       or iteration == self.n_impressions))

    if test_print:
      cur_results = self.evaluate_ranker(iteration,
                                         ranker,
                                         ranking_i,
                                         train_ranking,
                                         ranking_labels)
      self.online_score += cur_results['display']*self.cur_online_discount
      cur_results['cumulative-display'] = self.online_score
      results.append(cur_results)
    else:
      cur_results = self.evaluate_ranker_train_only(iteration,
                                                    ranker,
                                                    ranking_i,
                                                    train_ranking,
                                                    ranking_labels)
      self.online_score += cur_results['display']*self.cur_online_discount
      if self.print_all_train:
        cur_results['cumulative-display'] = self.online_score
        results.append(cur_results)

    self.cur_online_discount *= self.online_discount

    if iteration >= self.next_print:
      if self.print_logscale and iteration >= self.print_next_scale:
          self.print_next_scale *= self.print_scale
          self.print_frequency *= self.print_scale

      self.last_print = self.next_print
      self.next_print = self.next_print + self.print_frequency


  def evaluate_ranker(self, iteration, ranker,
                      ranking_i, train_ranking,
                      ranking_labels):

    test_rankings = ranker.get_test_rankings(
                    self.datafold.test_feature_matrix,
                    self.datafold.test_doclist_ranges,
                    inverted=True)
    test_ndcg = evaluate(
                  test_rankings,
                  self.datafold.test_label_vector,
                  self.test_idcg_vector,
                  self.datafold.test_doclist_ranges.shape[0] - 1,
                  self.n_results)

    train_ndcg = evaluate_ranking(
            train_ranking,
            ranking_labels,
            self.train_idcg_vector[ranking_i],
            self.n_results)

    results = {
      'iteration': iteration,
      'heldout': np.mean(test_ndcg),
      'display': np.mean(train_ndcg),
    }

    for name, value in ranker.get_messages().items():
      results[name] = value

    return results

  def evaluate_ranker_train_only(self, iteration, ranker,
                                 ranking_i, train_ranking,
                                 ranking_labels):

    train_ndcg = evaluate_ranking(
            train_ranking,
            ranking_labels,
            self.train_idcg_vector[ranking_i],
            self.n_results)

    results = {
      'iteration': iteration,
      'display': np.mean(train_ndcg),
    }

    for name, value in ranker.get_messages().items():
      results[name] = value

    return results

  def sample_and_rank(self, ranker):
    ranking_i = np.random.choice(self.datafold.n_train_queries())
    train_ranking = ranker.get_train_query_ranking(ranking_i)

    assert train_ranking.shape[0] <= self.n_results, 'Shape is %s' % (train_ranking.shape,)
    assert len(train_ranking.shape) == 1, 'Shape is %s' % (train_ranking.shape,)

    return ranking_i, train_ranking


  def run(self, ranker, output_key):
    starttime = time.time()

    ranker.setup(train_features = self.datafold.train_feature_matrix,
                 train_query_ranges = self.datafold.train_doclist_ranges)

    run_results = []
    impressions = 0

    text = open("Weights.txt","r")
    lines = text.read().split(',')
    lines = [float(i) for i in lines]
    attacker_weights = np.expand_dims(np.asarray(lines), axis=1)

    vector_norms = np.sum(attacker_weights ** 2, axis=0) ** (1. / 2)
    attacker_weights /= vector_norms[None, :]
    ndcgs = []
    ndcgs_attacker = []
    taus = []
    ndcg = 0
    ndcg_attacker = 0

    for impressions in range(self.n_impressions):
      # ranking_i, train_ranking = self.sample_and_rank(ranker)
      ranking_i = np.random.choice(self.datafold.n_train_queries())
      query_feat = ranker.get_query_features(ranking_i,
                                     ranker._train_features,
                                     ranker._train_query_ranges)

      doc_scores = ranker.model.score(query_feat)
      system_ranking = rnk.rank_query(doc_scores)
      freq = {}
      attacker_wait = 10

      while attacker_wait:

        train_ranking = ranker.get_train_query_ranking(ranking_i)
        for r in train_ranking:
          freq[r] = freq.get(r, 0) + 1

        attacker_wait -= 1

      ranking_labels = self.datafold.train_query_labels(ranking_i)

      # print train_ranking
      # print "============"
      should_click = []

      #Not required but kept for checking
      import operator
      sorted_freqs = sorted(freq.items(), key=operator.itemgetter(1), reverse=True)
      # print " Sorted version: "
      # print sorted_freqs

      for key in freq:
        if freq[key] >= 9:
          should_click.append(key)


      X = []
      start_doc = self.datafold.train_doclist_ranges[ranking_i]
      end_doc = self.datafold.train_doclist_ranges[ranking_i+1]
      train_docs = [id for id in range(start_doc, end_doc)]

      for id in train_docs:
        X.append([self.datafold.train_feature_matrix[id, :]])
   

      X = np.array(X)     # (10x1x41)
      attacker_scores = np.dot(X,attacker_weights)
      attacker_scores = attacker_scores/np.linalg.norm(attacker_scores)
  
      temp = [(attacker_scores[i-start_doc], i-start_doc)  for i in range(start_doc, end_doc)]
      temp = sorted(temp, key = lambda x: (-1*x[0]))
      attacker_ranking = list(map(lambda x: x[1], temp))


      # clicks = self.click_model.generate_clicks(train_ranking, ranking_labels)
      clicks = self.click_model.generate_mal_clicks(train_ranking, ranking_labels, attacker_scores, attacker_ranking, impressions)
      # clicks = self.click_model.generate_new_mal_clicks(train_ranking, attacker_ranking, should_click)

      test_r = ranker.get_test_rankings(self.datafold.test_feature_matrix, self.datafold.test_doclist_ranges, inverted=True)
      # test_ndcg = evaluate(
      #               test_r,
      #               self.datafold.test_label_vector,
      #               self.test_idcg_vector,
      #               self.datafold.test_doclist_ranges.shape[0] - 1,
      #               self.n_results)
      
      # print "Test : "
      # print self.datafold.test_feature_matrix.shape
      # print test_r.shape   =>   11742

      X = []
      for id in range(len(test_r)):
      	X.append([self.datafold.test_feature_matrix[id, :]])
      X = np.array(X)

      attacker_scores = np.dot(X,attacker_weights)
      attacker_scores = attacker_scores/np.linalg.norm(attacker_scores)

      initial = 0
      comb_tau = 0
      na = 0
      r = 0
      # attacker_list = []
    
      while(initial < self.datafold.test_doclist_ranges.shape[0]-1):
        start_doc = self.datafold.test_doclist_ranges[initial]
        end_doc = self.datafold.test_doclist_ranges[initial+1]
        test_labels = self.datafold.test_query_labels(initial)

        temp = [(attacker_scores[i], i-start_doc)  for i in range(start_doc, end_doc)]
        temp = sorted(temp, key = lambda x: (-1*x[0]))
        attacker_ranking = list(map(lambda x: x[1], temp))

        assert len(attacker_ranking) == test_r[start_doc:end_doc].shape[0]

        r = [0 for i in range(len(attacker_ranking))]
        for i in range(len(attacker_ranking)):
	    	if test_r[start_doc+i] < len(r):
	       		r[test_r[start_doc+i]] = i

        # tau, _ = stats.kendalltau(r, test_r[start_doc:end_doc])
        tau, _ = stats.kendalltau(r, attacker_ranking)
        comb_tau += tau

        # attacker_list.extend(attacker_ranking)
        # ndcg += compute_ndcg(attacker_ranking, test_r[start_doc:end_doc], 10)
        ndcg += get_ndcg_with_labels(test_r[start_doc:end_doc], test_labels, 10, initial, is_test = True)
        initial += 1
        na = get_ndcg_with_ranking2(test_r[start_doc:end_doc], attacker_ranking, 10, initial, impressions)
        ndcg_attacker += na

        # print " query: ", initial
        # print attacker_ranking
        # print r
        # print na
        # print "----------------------"

      # if impressions % 100 == 0:
	     #  # print "----------------------"
	     #  print 'Impression: ', impressions
	     #  print attacker_ranking
	     #  print r
	     #  print na
	     #  print "----------------------"

      if impressions % 10 == 0:
          ndcgs.append((ndcg/10.0)/initial)
          ndcgs_attacker.append((ndcg_attacker/10.0)/initial)
          ndcg = 0
          ndcg_attacker = 0

      taus.append(comb_tau/initial)
      # ndcgs.append(ndcg/initial)

      # print("my vs. ori", ndcg/initial, test_ndcg)
      # assert ndcg/initial == test_ndcg


      self.timestep_evaluate(run_results, impressions, ranker,
                             ranking_i, train_ranking, ranking_labels)

      ranker.process_clicks(clicks)

    # print ranker.model.weights
    # evaluate after final iteration
    x2 = [i for i in range(self.n_impressions-1)] 
    x = [i for i in range(self.n_impressions/10-1)]
    # plt.plot(x, taus)
    ndcgs = ndcgs[1:]
    fig_ndcg, ax_ndcg = plt.subplots()
    ax_ndcg.plot(x, ndcgs)
    ax_ndcg.set_xlabel("Impressions")
    ax_ndcg.set_ylabel(self.datafold.name + ": Fold " + str(self.datafold.fold_num+1) + " NDCGs")

    ndcgs_attacker = ndcgs_attacker[1:]
    fig_ndcg_attacker, ax_ndcg_attacker = plt.subplots()
    ax_ndcg_attacker.plot(x, ndcgs_attacker)
    ax_ndcg_attacker.set_xlabel("Impressions")
    ax_ndcg_attacker.set_ylabel(self.datafold.name + ": Fold " + str(self.datafold.fold_num+1) + " NDCGs attacker")

    taus = taus[1:]
    fig_ndcg, ax_ndcg = plt.subplots()
    ax_ndcg.plot(x2, taus)
    ax_ndcg.set_xlabel("Impressions")
    ax_ndcg.set_ylabel(self.datafold.name + ": Fold " + str(self.datafold.fold_num+1) + " Tau")
    plt.show()

    ranking_i, train_ranking = self.sample_and_rank(ranker)
    ranking_labels =  self.datafold.train_query_labels(ranking_i)
    impressions += 1
    self.timestep_evaluate(run_results, impressions, ranker,
                           ranking_i, train_ranking, ranking_labels)

    ranker.clean()

    self.run_details['runtime'] = time.time() - starttime

    output = {'run_details': self.run_details,
              'run_results': run_results}

    self.output_queue.put((output_key, output))


# Custom NDCG function written by Rishab (Considering binary relevance only)
def compute_ndcg(attacker_ranking, model_ranking, k):

    num = 0
    denom = 0
    i = 1
    for r in model_ranking:
      if i > k:
          break
      if r in attacker_ranking[0:k]:
          num += 1/(math.log(1+i, 2))
      i += 1

    for j in range(1,k+1):
      if j > len(attacker_ranking):
        break
      denom += 1/(math.log(1+j, 2))

    return num/denom
