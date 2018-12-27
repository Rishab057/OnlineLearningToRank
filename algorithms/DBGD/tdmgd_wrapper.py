

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from models.linearmodel import LinearModel
import utils.rankings as rnk
from algorithms.DBGD.tddbgd import TD_DBGD

# Dueling Bandit Gradient Descent
class TD_MGD_Wrapper(TD_DBGD):

  def __init__(self, svd, project_norm, k_initial, k_increase, n_candidates, _lambda=None, *args, **kargs):
    super(TD_MGD_Wrapper, self).__init__(*args, **kargs)
    self.model = LinearModel(n_features = self.n_features,
                             learning_rate = self.learning_rate,
                             n_candidates = n_candidates)

    self.svd = svd
    self.project_norm = project_norm
    self.k_initial = k_initial
    self.k_increase = k_increase
    self._lambda = _lambda
    # self.n_results = 60

  @staticmethod
  def default_parameters():
    parent_parameters = TD_DBGD.default_parameters()
    parent_parameters.update({
      'n_candidates': 9
      })
    return parent_parameters

  def update_to_interaction(self, clicks):
    winners = self.multileaving.winning_rankers(clicks)
    ###############################################################
    if True in clicks:
      # For projection
      # keep track of feature vectors of doc list
      viewed_list = []
      # index of last click
      last_click = max(loc for loc, val in enumerate(clicks) if val == True)
      # prevent last_click+k from exceeding interleaved list length
      k_current = self.k_initial
      if self.k_increase:
        # gradually increast k
        k_current += int(self.n_interactions/1000)
      last_doc_index = min(last_click+k_current, len(self._last_ranking)-1)
      # print(last_doc_index)

      query_feat = self.get_query_features(self.query_id,
                                       self._train_features,
                                       self._train_query_ranges)
      for i in range(last_doc_index):
        docid = self._last_ranking[i]
        feature = query_feat[docid]
        viewed_list.append(feature)
      self.model.update_to_mean_winners(winners,viewed_list,self.svd,self.project_norm)
    ###############################################################
    else:
      self.model.update_to_mean_winners(winners)


  def _create_train_ranking(self, query_id, query_feat, inverted):
    # Save query_id to get access to query_feat when updating
    self.query_id = query_id
    assert inverted == False
    self.model.sample_candidates()
    scores = self.model.candidate_score(query_feat)
    rankings = rnk.rank_single_query(scores, inverted=False, n_results=self.n_results)
    multileaved_list = self.multileaving.make_multileaving(rankings)
    return multileaved_list