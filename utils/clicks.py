import numpy as np
import operator
import random


class ClickModel(object):

  '''
  Class for cascading click-models used to simulate clicks.
  '''

  def __init__(self, name, data_type, PCLICK, PSTOP):
    '''
    Name is used for logging, data_type denotes the degrees of relevance the data uses.
    PCLICK and PSTOP the probabilities used by the model.
    '''
    self.name = name
    self.type = data_type
    self.PCLICK = PCLICK
    self.PSTOP = PSTOP

  def get_name(self):
    '''
    Name that can be used for logging.
    '''
    return self.name + '_' + self.type

  def generate_clicks(self, ranking, all_labels):
    '''
    Generates clicks for a given ranking and relevance labels.
    ranking: np array of indices which correspond with all_labels
    all_labels: np array of integers
    '''
    labels = all_labels[ranking]
    coinflips = np.random.rand(*ranking.shape)
    clicks = coinflips < self.PCLICK[labels]
    coinflips = np.random.rand(*ranking.shape)
    stops = coinflips < self.PSTOP[labels]
    stopped_clicks = np.zeros(ranking.shape, dtype=bool)
    if np.any(stops):
        clicks_before_stop = np.logical_and(clicks, np.arange(ranking.shape[0])
                                            <= np.where(stops)[0][0])
        stopped_clicks[clicks_before_stop] = True
        return stopped_clicks
    else:
        return np.zeros(ranking.shape, dtype=bool) + clicks


class ExamineClickModel(object):

  '''
  Class for cascading click-models used to simulate clicks.
  '''

  def __init__(self, name, data_type, PCLICK, eta):
    '''
    Name is used for logging, data_type denotes the degrees of relevance the data uses.
    PCLICK and PSTOP the probabilities used by the model.
    '''
    self.name = name
    self.type = data_type
    self.PCLICK = PCLICK
    self.eta = eta

  def get_name(self):
    '''
    Name that can be used for logging.
    '''
    return self.name + '_' + self.type

  def generate_clicks(self, ranking, all_labels):
    '''
    Generates clicks for a given ranking and relevance labels.
    ranking: np array of indices which correspond with all_labels
    all_labels: np array of integers
    '''
    n_results = ranking.shape[0]
    examine_prob = (1./(np.arange(n_results)+1))**self.eta
    stop_prob = np.ones(n_results)
    stop_prob[1:] -= examine_prob[1:]/examine_prob[:-1]
    stop_prob[0] = 0.

    labels = all_labels[ranking]
    coinflips = np.random.rand(*ranking.shape)
    clicks = coinflips < self.PCLICK[labels]
    coinflips = np.random.rand(n_results)
    stops = coinflips < stop_prob
    stops = np.logical_and(stops, clicks)
    stopped_clicks = np.zeros(ranking.shape, dtype=bool)
    if np.any(stops):
        clicks_before_stop = np.logical_and(clicks, np.arange(ranking.shape[0])
                                            <= np.where(stops)[0][0])
        stopped_clicks[clicks_before_stop] = True
        return stopped_clicks
    else:
        return np.zeros(ranking.shape, dtype=bool) + clicks

class MaliciousClickModel(object):

  '''
  Class for cascading click-models used to simulate malicious clicks.
  '''

  def __init__(self, name, data_type):
    '''
    Name is used for logging and identifying the attack type.
    '''
    self.name = name
    self.type = data_type

  def get_name(self):
    '''
    Name that can be used for logging.
    '''
    return self.name + '_' + self.type

  def generate_clicks(self, train_ranking, attacker_ranking, start, end, freq, mf, sd_const):
      
      if self.name == "naive_intersection_attack":
        return self.naive_intersection_attack(train_ranking, attacker_ranking, start, end)
      elif self.name == "frequency_attack":
        return self.frequency_attack(train_ranking, attacker_ranking, freq, mf, start, end, sd_const)
      else:
        print("Attack name is incorrect. Only 'naive_intersection_attack' and 'frequency_attack' are supported!!\n")


  def naive_intersection_attack(self, train_ranking, attacker_ranking, start, end):
    '''
    Generates malicious clicks based on the intersection of train_ranking and attacker_ranking.
    Intersection is guided by start and end hyper-parameters.
    '''
    clicks = []

    for i in range(0, len(train_ranking)):

      if (len(attacker_ranking) >= end and train_ranking[i] in attacker_ranking[start:end]):
        clicks.append(True)
      else:
        clicks.append(False)

    return np.zeros(train_ranking.shape, dtype=bool) + clicks


  def frequency_attack(self, train_ranking, attacker_ranking, freq, mf, start, end, sd_const):
    '''
    Generates malicious clicks based on the intersection of train_ranking and attacker_ranking.
    Intersection is guided by start and end hyper-parameters.
    mf controls which documents get clicked in the intersection.
    mf: Number of most frequent docs that the attacker assumes come from the current ranker.
    freq: Frequency table containing (doc, freq)
    '''
    
    # Sorting the frequency based on frequency
    sorted_freqs = sorted(freq.items(), key=operator.itemgetter(1), reverse=True)

    # Breaking the table into doc and frequency. The top_k_docs can be considered as a proxy for current ranker's ranking
    i = 0
    top_k_docs = []
    top_k_freq = []
    while (i < len(sorted_freqs) and i<mf):
        top_k_docs.append(sorted_freqs[i][0])
        top_k_freq.append(sorted_freqs[i][1])
        i += 1

    clicks = []

    # Finding the standard deviation of top-k frequency list
    sd = np.std(top_k_freq) if (len(top_k_freq) > 0) else 0

    # Finding the index of position which is atleast sd_const standard deviations away. If such position is not found then mf will be over-rided otherwise not.
    ind = len(top_k_freq) 
    for index in range(0, len(top_k_freq)-1):
        if sd_const*sd <= top_k_freq[index] - top_k_freq[index+1]:
            ind = index+1
            break;

    # Generating the clicks using the intersection and also making sure that the document is not one of the most frequent documents
    for i in range(0, len(train_ranking)):
      if train_ranking[i] not in top_k_docs[0:ind] and train_ranking[i] in attacker_ranking[start:end]:
        clicks.append(True)
      else:
        clicks.append(False)

    return np.zeros(train_ranking.shape, dtype=bool) + clicks


# create synonyms for keywords to ease command line use
syn_tuples = [
    ('ex_per_1', ['exper1']),
    ('navigational', ['nav', 'navi', 'navig', 'navigat']),
    ('informational', ['inf', 'info', 'infor', 'informat']),
    ('perfect', ['per', 'perf']),
    ('almost_random', [
        'alm',
        'almost',
        'alra',
        'arand',
        'almostrandom',
        'almrand',
        ]),
    ('random', ['ran', 'rand']),
    ('binary', ['bin']),
    ('short', []),
    ('long', []),
    ]
attack_tuples = [
    ('naive_intersection_attack', []),
    ('frequency_attack', []),
]
synonyms = {}
for full, abrv_list in syn_tuples:
    assert full not in synonyms or synonyms[full] == full
    synonyms[full] = full
    for abrv in abrv_list:
        assert abrv not in synonyms or synonyms[abrv] == full
        synonyms[abrv] = full

attack_synonyms = {}
for full, abrv_list in attack_tuples:
    assert full not in attack_synonyms or attack_synonyms[full] == full
    attack_synonyms[full] = full
    for abrv in abrv_list:
        assert abrv not in attack_synonyms or attack_synonyms[abrv] == full
        attack_synonyms[abrv] = full

bin_models = {}
bin_models['navigational'] = np.array([.05, .95]), np.array([.2, .9])
bin_models['informational'] = np.array([.4, .9]), np.array([.1, .5])
bin_models['perfect'] = np.array([.0, 1.]), np.array([.0, .0])
bin_models['almost_random'] = np.array([.4, .6]), np.array([.5, .5])
bin_models['random'] = np.array([.5, .5]), np.array([.0, .0])
bin_models['ex_per_1'] = np.array([.0, 1.]), 1.0

short_models = {}
short_models['navigational'] = np.array([.05, .5, .95]), np.array([.2, .5, .9])
short_models['informational'] = np.array([.4, .7, .9]), np.array([.1, .3, .5])
short_models['perfect'] = np.array([.0, .5, 1.]), np.array([.0, .0, .0])
short_models['almost_random'] = np.array([.4, .5, .6]), np.array([.5, .5, .5])
short_models['random'] = np.array([.5, .5, .5]), np.array([.0, .0, .0])
short_models['ex_per_1'] = np.array([.0, .5, 1.]), 1.0

long_models = {}
long_models['navigational'] = np.array([.05, .3, .5, .7, .95]), np.array([.2, .3, .5, .7, .9])
long_models['informational'] = np.array([.4, .6, .7, .8, .9]), np.array([.1, .2, .3, .4, .5])
long_models['perfect'] = np.array([.0, .2, .4, .8, 1.]), np.array([.0, .0, .0, .0, .0])
long_models['almost_random'] = np.array([.4, .45, .5, .55, .6]), np.array([.5, .5, .5, .5, .5])
long_models['random'] = np.array([.5, .5, .5, .5, .5]), np.array([.0, .0, .0, .0, .0])
long_models['ex_per_1'] = np.array([.0, .2, .4, .8, 1.]), 1.0

all_models = {'short': short_models, 'binary': bin_models, 'long': long_models}

def get_click_models(keywords):
    '''
  Convenience function which returns click models corresponding with keywords.
  only returns click functions for one data type: (bin,short,long)
  '''
    type_name = None
    type_keyword = None
    # print("Keywords: ", keywords)
    for keyword in keywords:
        assert (keyword in synonyms) or (keyword in attack_synonyms)
        if keyword in synonyms and synonyms[keyword] in all_models:
            type_name = synonyms[keyword]
            type_keyword = keyword
            break
    assert type_name is not None and type_keyword is not None

    models_type = all_models[type_name]
    full_names = []
    for key in keywords:
        if key in synonyms and key != type_keyword:
            full_names.append(synonyms[key])
        if key in attack_synonyms:
            full_names.append(attack_synonyms[key])

    click_models = []

    for full in full_names:
        if full in attack_synonyms:
            c_m = MaliciousClickModel(full, type_name)
        elif full == 'ex_per_1':
            c_m = ExamineClickModel(full, type_name, *models_type[full])
        else:
            c_m = ClickModel(full, type_name, *models_type[full])
        click_models.append(c_m)

    return click_models