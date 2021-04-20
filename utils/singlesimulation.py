# -*- coding: utf-8 -*-

import time
import numpy as np
from evaluate import *
from clicks import *
import scipy.stats as stats # For kendall tau
import matplotlib.pyplot as plt
import math
import json # FOr reading queue.txt
import time
import random


class SingleSimulation(object):

  def __init__(self, sim_args, output_queue, click_model, datafold):
    self.train_only = sim_args.train_only
    self.n_impressions = sim_args.n_impressions
    self.n_results = sim_args.n_results
    self.click_model = click_model
    self.datafold = datafold

    #  Extra functionalities added by Rishab
    self.mf = sim_args.mf
    self.sd_const = sim_args.sd_const
    self.which = sim_args.which
    self.start = sim_args.start
    self.end = sim_args.end
    self.user_click_model = sim_args.user_click_model

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


  def run(self, ranker, output_key, attacker_output_key):
    starttime = time.time()

    if "frequency" in self.click_model.name:
      print "Name: ", self.click_model.name, " n_res:", self.n_results, " start:", self.start, " end:", self.end, " mf:", self.mf, " sd:", self.sd_const

    else:
      print "Name: ", self.click_model.name, " n_res:", self.n_results, " start:", self.start, " end:", self.end

    ranker.setup(train_features = self.datafold.train_feature_matrix,
                 train_query_ranges = self.datafold.train_doclist_ranges)


    #Get the normal user click model, and the attacker weights
    normal_click_model = get_click_models([self.user_click_model] + [self.datafold.click_model_type])[0]
    attacker_weights = get_attacker_weights(self.datafold.name)

    run_results = []
    attacker_results = []
    iteration = 0
    noc = 0
    noac = []
    queries_attacked = 0
    queries_attacked_per_1000 = 0
    noexpl_total = 0

    repeat = 0
    winners = 0
    repeta = []
    cosine_sim = []
    cosine_sim_curr = []
    cosine_sim_neg = []
    cosine_sim_curr_neg = []


    for impressions in range(self.n_impressions):

      #Get the train ranking 
      ranking_i, train_ranking = self.sample_and_rank(ranker)


      #Get the training features  
      train_feat = ranker.get_query_features(ranking_i,
                                       self.datafold.train_feature_matrix,
                                       self.datafold.train_doclist_ranges)

      #Get the associated ground-truth labels
      ranking_labels = self.datafold.train_query_labels(ranking_i)

      #Find the attacker ranking
      attacker_ranking = get_attacker_ranking(train_feat, attacker_weights)

      
      clicks = np.zeros(train_ranking.shape, dtype=bool)

      # Find whether attack needs to be done in this iteration or not
      attack = False
      if(self.which == 0):
        attack = True
      elif(self.which == 1):
        attack = (iteration <= 2000) 
      elif(self.which == 2):
        attack = (iteration > 2000 and iteration <= 4000)
      elif(self.which == 3):
        attack = (iteration > 4000 and iteration <= 6000)
      elif(self.which == 4):
        attack = (iteration > 6000 and iteration <= 8000)
      elif(self.which == 5):
        attack = (iteration > 8000 and iteration <= 10000)
      else:
        attack = False

      if attack:
        # If attack needs to be done, find which attack

        freq = {}
        if self.click_model.name == "frequency_attack":
          # If it is frequency attack then repeat the same query 9 times
          for i in range(0, 9):
            if iteration > self.n_impressions:
              break;
            if i > 0:
              train_ranking = ranker.get_train_query_ranking(ranking_i)

            # Create/Update the frequency table (freq)
            for r in train_ranking:
              freq[r] = freq.get(r, 0) + 1
            
            # Sort the frequency in decreasing order of frequency
            sorted_freqs = sorted(freq.items(), key=operator.itemgetter(1), reverse=True)

            # Create a top-10 frequent list
            temp = 10
            ik = 0
            top_k = []

            while (ik < len(sorted_freqs) and ik<temp):
                  top_k.append(sorted_freqs[ik][1])
                  ik += 1

            # Find standard deviation of the top-10 frequent list
            sd = np.std(top_k)

            if(i < 8):
              # Doing evaluation in every iteration even when no clicks are happening

              self.test_evaluation(attacker_results, iteration, ranker, clicks, self.n_results, ranker.model.learning_rate, attacker_weights)
              self.timestep_evaluate(run_results, iteration, ranker, ranking_i, train_ranking, ranking_labels)
              ranker.process_clicks(clicks)
              iteration += 1

            # If the sd > 1, then find the position which is atleast sd_cont standard positions away from the next position. If found break the loop.
            if sd > 1:
              found = False 
              for index in range(0, len(top_k)-1):
                if self.sd_const*sd <= top_k[index] - top_k[index+1]:
                  found = True
                  break;
              if found:
                break;
      
        teams = ranker.multileaving.teams
        # Generating malicious clicks
        clicks, noexpl = self.click_model.generate_clicks(train_ranking, attacker_ranking, teams, self.start, self.end, freq, self.mf, self.sd_const)
        noexpl_total += noexpl

      else:
        clicks = normal_click_model.generate_clicks(train_ranking, ranking_labels)

      if iteration <= self.n_impressions:
        self.test_evaluation(attacker_results, iteration, ranker, clicks, self.n_results, ranker.model.learning_rate, attacker_weights)       
        self.timestep_evaluate(run_results, iteration, ranker,
                               ranking_i, train_ranking, ranking_labels)

        ranker.process_clicks(clicks)
        noc += np.count_nonzero(clicks)
        iteration += 1

    ranking_i, train_ranking = self.sample_and_rank(ranker)
    ranking_labels =  self.datafold.train_query_labels(ranking_i)
    self.timestep_evaluate(run_results, iteration, ranker,
                           ranking_i, train_ranking, ranking_labels)

    ranker.clean()

    self.run_details['runtime'] = time.time() - starttime

    output = {'run_details': self.run_details,
              'run_results': run_results}

    attacker_output = {'run_details': self.run_details,
                       'run_results': attacker_results}


    self.output_queue.put((attacker_output_key, attacker_output))
    self.output_queue.put((output_key, output))



  def test_evaluation(self, attacker_results, iteration, ranker, clicks, n_results, lr, attacker_weights):

      # Getting the test_ranking in inverted format.
      test_r = ranker.get_test_rankings(self.datafold.test_feature_matrix, self.datafold.test_doclist_ranges, inverted=True)

      ndcg_attack, ndcg_label, tau, tau_sum = 0, 0, 0, 0
      for test_query in range(self.datafold.test_doclist_ranges.shape[0]-1):
        # Find the start doc and end doc from the ranges
        start_doc = self.datafold.test_doclist_ranges[test_query]
        end_doc = self.datafold.test_doclist_ranges[test_query+1]

        # Find the ground truth test labels
        test_labels = self.datafold.test_query_labels(test_query)

        # Get the test features and create the attacker ranking
        test_features = self.datafold.test_feature_matrix[start_doc:end_doc, :]
        attacker_ranking = get_attacker_ranking(test_features, attacker_weights)

        assert len(attacker_ranking) == test_r[start_doc:end_doc].shape[0]

        # Find the NDCG performance wrt. Attacker and wrt. Gound Truth 
        ndcg_attack += get_ndcg_with_ranking(test_r[start_doc:end_doc], attacker_ranking, n_results)
        ndcg_label += get_ndcg_with_labels(test_r[start_doc:end_doc], test_labels, n_results)
        tau, _ = stats.kendalltau(attacker_ranking, test_r[start_doc:end_doc])
        tau_sum += tau

      num_clicks = np.count_nonzero(clicks)

      cur_results = {
        'iteration': iteration,
        'NDCG_attack': ndcg_attack/test_query,
        'NDCG_label': ndcg_label/test_query,
        'Kendall\'s Tau': tau_sum/test_query,
        'Click Number': num_clicks,
        'LR': lr,
      }

      for name, value in ranker.get_messages().items():
        cur_results[name] = value
      attacker_results.append(cur_results)


def get_attacker_ranking(features, attacker_weights):
    '''
    Given features and weights get the attacker ranking. 
    '''

    # Finding scores by doing dot product
    attacker_scores = np.dot(features, attacker_weights)

    # Normalizing the scores
    norm_value = np.linalg.norm(attacker_scores)
    if norm_value > 0:
      attacker_scores = attacker_scores/norm_value

    # Finding the ranking by first creating the (score, doc) pair and then sorting by score in decreasing order
    attacker_score_document_pair = [(attacker_scores[i], i) for i in range(len(attacker_scores))]
    attacker_score_document_pair = sorted(attacker_score_document_pair, key = lambda x: (-1*x[0], x[1]))
    attacker_ranking = list(map(lambda x: x[1], attacker_score_document_pair))
    return attacker_ranking

def get_attacker_weights(name):
    '''
    Get the attacker weights from the given dataset name.
    '''
    if "MSLR" in name:
      attacker_weights_file = open("Weights_web10k.txt","r")
    elif "MQ2007" in name:
      attacker_weights_file = open("Weights_mq2007.txt","r")
    elif "Webscope" in name:
      attacker_weights_file = open("Weights_yahoo.txt","r")
    elif "2003" in name:
      attacker_weights_file = open("Weights_td2003.txt","r")
    else:
      print("Weight file not specified")

    attacker_weights_lines = attacker_weights_file.read().split(',')
    attacker_weights_lines = [float(i) for i in attacker_weights_lines]
    attacker_weights = np.expand_dims(np.asarray(attacker_weights_lines), axis=1)

    return attacker_weights