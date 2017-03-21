from __future__ import print_function
from itertools import  product
import pprint
#ignore warnings for clear printing
import warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer , TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
import pandas as pd
from sklearn.cluster import *
import logging
import sys
import os
from time import time
import numpy as np

##########Import data##############
indir = './book_texts/one_line_text'
books_names = []
books_texts = []
for texts in os.listdir(indir):
    name, ex = os.path.splitext(texts)
    books_names.append(name)
    books_texts.append(open(indir + "/" + texts).read())
print(books_names)

real_names= {'gap_-C0BAAAAQAAJ' : 'Dictionary of Greek and Roman Geography, Edited by William Smith, Vol II',
'gap_2X5KAAAAYAAJ' : 'The Works of Cornelius Tacitus, by Arthur Murphy, Vol V',
'gap_9ksIAAAAQAAJ' : 'The History of the Peloponnesian War, translated from the Greek of Thucydides, By William Smith, Vol II',
'gap_Bdw_AAAAYAAJ' : 'The History of Rome By Titus Liviusm, translated by George Baker, Vol I',
'gap_CSEUAAAAYAAJ' : 'The History of the Decline and Fall of The Roman Empire, by Edward Gibbon, Vol III',
'gap_CnnUAAAAMAAJ' : 'The Whole Genuie Works of Flavius Josephus, by William Whiston, Vol II',
'gap_DhULAAAAYAAJ' : 'The Description of Greece, by Pausanias, Vol III',
'gap_DqQNAAAAYAAJ' : 'LIVY , History of Rome, translated by George Baker, Vol III',
'gap_GIt0HMhqjRgC' : 'Gibbon\'s History of the Decline and Fall of The Roman Empire, by Thomas Bowdler, Vol IV',
'gap_IlUMAQAAMAAJ' : 'Gibbon\'s History of the Decline and Fall of the Roman Empire, by Thomas Bowdler, Vol II',
'gap_MEoWAAAAYAAJ' : 'The Historical Annals of Cornelius Tacitus, by Arthur Murphy, Vol I',
'gap_RqMNAAAAYAAJ' : 'LIVY , History of Rome, translated by George Baker, Vol V',
'gap_TgpMAAAAYAAJ' : 'The Genuie Works of Flavius Josephus, by William Whiston, Vol I',
'gap_VPENAAAAQAAJ' : 'The History of the Decline and Fall of The Roman Empire, by Edward Gibbon, Vol V',
'gap_WORMAAAAYAAJ' : 'The Histories of Caius Cornelius Tacitus, by William Seymour Tyler',
'gap_XmqHlMECi6kC' : 'The History of the Decline and Fall of The Roman Empire, by Edward Gibbon, Vol VI',
'gap_aLcWAAAAQAAJ' : 'The History of the Decline and Fall of The Roman Empire, by Edward Gibbon, Vol I',
'gap_dIkBAAAAQAAJ' : 'The History of Rome by Theoder Mommsen, translated by William P. Dickson, Vol III',
'gap_fnAMAAAAYAAJ' : 'The History of the Peloponnesian War by Thucydides, By William Smith, Vol I',
'gap_m_6B1DkImIoC' : 'Titus Livus, Roman History, by William Gordon',
'gap_ogsNAAAAIAAJ' : 'The Works of Josephus, by William Whiston, Vol IV',
'gap_pX5KAAAAYAAJ' : 'The Works of Cornelius Tacitus, by Arthur Murphy, Vol IV',
'gap_udEIAAAAQAAJ' : 'The First and Thirty-Third Books of Pliny\'s Natural History, by John Bostock',
'gap_y-AvAAAAYAAJ' : 'The Genuie Works of Flavius Josephus, by William Whiston, Vol III'}

#######desirable outcome labels##############
labels7_1 = [6, 2, 6, 6, 0, 5, 5, 2, 1, 0, 5, 1, 4, 0, 6, 5, 0, 1, 5, 4, 5, 3, 1, 2]
labels7_2 = [6, 2, 6, 6, 0, 5, 5, 2, 1, 0, 5, 1, 4, 0, 6, 5, 0, 1, 5, 4, 5, 3, 1, 4]
labels8_1 = [7, 2, 7, 7, 0, 6, 6, 2, 1, 0, 6, 1, 5, 0, 7, 6, 0, 1, 6, 5, 6, 4, 1, 3]
labels8_2 = [7, 2, 7, 7, 0, 6, 6, 2, 1, 0, 6, 1, 3, 0, 7, 6, 0, 1, 6, 5, 6, 4, 1, 2]
labels9_1 = [8, 2, 8, 8, 0, 7, 7, 2, 1, 0, 7, 1, 4, 0, 8, 7, 0, 1, 7, 6, 7, 5, 1, 3]

#### names of the metrics used###########
metrics_namesz = ['silhouette_score', 'calinski_harabaz_score']
metrics_names = ['adjusted_rand_score',
                 'normalized_mutual_info_score',
                 'mutual_info_score',
                 'adjusted_mutual_info_score',
                 'homogeneity_score', 'completeness_score',
                 'v_measure_score',
                 #'homogeneity_completeness_v_measure',
                 'fowlkes_mallows_score',
                 'silhouette_score',
                 'calinski_harabaz_score']

####metrics resullts dic############
##definitnion method
def define_metrics(dic_metrics):
    non_sum_labels = []
    for label in dic_metrics:
      labels_dict = {}
      if label != 'summary':
        non_sum_labels.append(label)
        for metrical in metrics_names:
          label_dict = {metrical: { 'val' : 0, 'res_label': [],  'vector_cfg': '', 'cluster_cfg': '', 'other_metrics': []}}
          labels_dict.update(label_dict)
        dic_metrics[label] = labels_dict

    summary_dict = {}
    for label in non_sum_labels:
      labels_dict = {}
      for metrical in metrics_names:
        label_dict = {metrical: { 'val' : 0, 'res_label': []}}
        labels_dict.update(label_dict)
      summary_dict[label] = labels_dict
    dic_metrics['summary'] = summary_dict

##definitnion method
def define_metrics_new(dic_metrics, label):
  labels_dict = {}
  for metrical in metrics_namesz:
    label_dict = {metrical: { 'val' : 0, 'res_label': [],  'vector_cfg': '', 'cluster_cfg': '', 'other_metrics': []}}
    labels_dict.update(label_dict)
  dic_metrics[label] = labels_dict

  summary_dict = dic_metrics['summary']
  labels_dict = {}
  for metrical in metrics_namesz:
    label_dict = {metrical: { 'val' : 0, 'res_label': []}}
    labels_dict.update(label_dict)
  summary_dict[label] = labels_dict
  dic_metrics['summary'] = summary_dict

##dictionaries
AggloClu_metrics = {'labels7_1': {}, 'labels7_2': {}, 'labels8_1': {}, 'labels8_2': {},  'labels9_1': {}, 'summary': {}}
KMeans_metrics = {'labels7_1': {}, 'labels7_2': {}, 'labels8_1': {}, 'labels8_2': {},  'labels9_1': {}, 'summary': {}}
Birch_metrics = {'labels7_1': {}, 'labels7_2': {}, 'labels8_1': {}, 'labels8_2': {},  'labels9_1': {}, 'summary': {}}
MeanShift_metrics = {'labels7_1': {}, 'labels7_2': {}, 'labels8_1': {}, 'labels8_2': {},  'labels9_1': {}, 'summary': {}}
AffinityPropagation_metrics = {'labels7_1': {}, 'labels7_2': {}, 'labels8_1': {}, 'labels8_2': {},  'labels9_1': {}, 'summary': {}}
common_metrics = {'labels7_1': {}, 'labels7_2': {}, 'labels8_1': {}, 'labels8_2': {},  'labels9_1': {}, 'summary': {}}
define_metrics(AggloClu_metrics)
define_metrics(KMeans_metrics)
define_metrics(Birch_metrics)
define_metrics(MeanShift_metrics)
define_metrics(AffinityPropagation_metrics)
define_metrics(common_metrics)

def print_res():
    print('############################################################################################')
    print('################# Final reults are      AggloClu_metrics####################################')
    print('############################################################################################')
    pprint.pprint(AggloClu_metrics, width=1)
    print('############################################################################################')
    print('################# Final reults are      KMeans_metrics####################################')
    print('############################################################################################')
    pprint.pprint(KMeans_metrics, width=1)
    print('############################################################################################')
    print('################# Final reults are       Birch_metrics####################################')
    print('############################################################################################')
    pprint.pprint(Birch_metrics, width=1)
    print('############################################################################################')
    print('################# Final reults are       MeanShift_metrics####################################')
    print('############################################################################################')
    pprint.pprint(MeanShift_metrics, width=1)
    print('############################################################################################')
    print('################# Final reults are       AffinityPropagation####################################')
    print('############################################################################################')
    pprint.pprint(AffinityPropagation_metrics, width=1)
    print('############################################################################################')
    print('################# Final reults are    Common Metrics####################################')
    print('############################################################################################')
    pprint.pprint(common_metrics, width=1)

####helper printer method########
def get_results_with_n(result_lab, n_clusters, X, best_metrics , vector_cfg, cluster_cfg):
    if   n_clusters == 7:
        target_labs = [labels7_1,labels7_2]
    elif n_clusters == 8:
        target_labs = [labels8_1,labels8_2]
    elif n_clusters == 9:
        target_labs = [labels9_1]

    for ind, target_lab in enumerate(target_labs):
        current_label = 'labels' + str(n_clusters) + '_' +  str(ind+1)
        print('@@@metric results for %s' % current_label)
        metrics_num = [metrics.adjusted_rand_score(target_lab, result_lab), #perect fit is 1.0
                      metrics.normalized_mutual_info_score(target_lab, result_lab),  #perfect fit is 1.0
                      metrics.mutual_info_score(target_lab, result_lab), #perfect fit is 1.0
                      metrics.adjusted_mutual_info_score(target_lab, result_lab), #perfect fit is 1.0
                      metrics.homogeneity_score(target_lab, result_lab), #perfect fit is 1.0
                      metrics.completeness_score(target_lab, result_lab), #perfect fit is 1.0
                      metrics.v_measure_score(target_lab, result_lab), #perfect fit is 1.0
                      #metrics.homogeneity_completeness_v_measure(target_lab, result_lab),
                      metrics.fowlkes_mallows_score(target_lab, result_lab), #perfect fit is 1.0
                      metrics.silhouette_score(X.todense(), result_lab, metric='euclidean'), #the higher the better
                      metrics.calinski_harabaz_score(X.todense(), result_lab)] #the higher the better

        print([metrics_names[index] +": "+ str(value) for index, value in enumerate(metrics_num)])
        for ind, metric in enumerate(metrics_num):
            ###save results for this clustering method
            if best_metrics[current_label][metrics_names[ind]]['val'] < metric:
                ###save summary##
                best_metrics['summary'][current_label][metrics_names[ind]]['val'] = metric
                best_metrics['summary'][current_label][metrics_names[ind]]['res_label'] = result_lab
                ###save comperhansive info####
                best_metrics[current_label][metrics_names[ind]]['val'] = metric
                best_metrics[current_label][metrics_names[ind]]['res_label'] = result_lab
                best_metrics[current_label][metrics_names[ind]]['vector_cfg'] = vector_cfg
                best_metrics[current_label][metrics_names[ind]]['cluster_cfg'] = cluster_cfg
                best_metrics[current_label][metrics_names[ind]]['other_metrics'] = [metrics_names[index] +": "+ str(value) for index, value in enumerate(metrics_num) if value != metric]
            ##save results for the co0mmon metrics
            if common_metrics[current_label][metrics_names[ind]]['val'] < metric:
                ###save summary##
                common_metrics['summary'][current_label][metrics_names[ind]]['val'] = metric
                common_metrics['summary'][current_label][metrics_names[ind]]['res_label'] = result_lab
                ###save comperhansive info####
                common_metrics[current_label][metrics_names[ind]]['val'] = metric
                common_metrics[current_label][metrics_names[ind]]['res_label'] = result_lab
                common_metrics[current_label][metrics_names[ind]]['vector_cfg'] = vector_cfg
                common_metrics[current_label][metrics_names[ind]]['cluster_cfg'] = cluster_cfg
                common_metrics[current_label][metrics_names[ind]]['other_metrics'] = [metrics_names[index] +": "+ str(value) for index, value in enumerate(metrics_num) if value != metric]

def get_results_without_n(result_lab, n_clusters, X, best_metrics , vector_cfg, cluster_cfg):

    if (n_clusters) > 6 and (n_clusters < 10):
        get_results_with_n(result_lab, n_clusters, X, best_metrics , vector_cfg, cluster_cfg)
        return;

    current_label = 'labels' + str(n_clusters) + '_1'
    print('@@@metric results for %s' % current_label)
    metrics_numz = [metrics.silhouette_score(X.todense(), result_lab, metric='euclidean'), metrics.calinski_harabaz_score(X.todense(), result_lab)]

    print([metrics_namesz[index] +": "+ str(value) for index, value in enumerate(metrics_numz)])
    for ind, metric in enumerate(metrics_numz):
        ###save results for this clustering method
        if best_metrics.has_key(current_label) == False:
            define_metrics_new(best_metrics,current_label)

        if best_metrics[current_label][metrics_namesz[ind]]['val'] < metric:
            ###save summary##
            best_metrics['summary'][current_label][metrics_namesz[ind]]['val'] = metric
            best_metrics['summary'][current_label][metrics_namesz[ind]]['res_label'] = result_lab
            ###save comperhansive info####
            best_metrics[current_label][metrics_namesz[ind]]['val'] = metric
            best_metrics[current_label][metrics_namesz[ind]]['res_label'] = result_lab
            best_metrics[current_label][metrics_namesz[ind]]['vector_cfg'] = vector_cfg
            best_metrics[current_label][metrics_namesz[ind]]['cluster_cfg'] = cluster_cfg
            best_metrics[current_label][metrics_namesz[ind]]['other_metrics'] = [metrics_namesz[index] +": "+ str(value) for index,value in enumerate(metrics_numz) if value != metric]

###AgglomerativeClustering parameters and definitoon####
AggloClu_n_clusters = [7,8,9]
AggloClu_affinity = ['euclidean' ,'l1', 'l2', 'manhattan', 'cosine', 'precomputed']
AggloClu_linkage = ['ward', 'complete', 'average']
def run_AggloClu(X, vector_cfg):
###run clustering
    for n_clusters, affinity, linkage in product(AggloClu_n_clusters, AggloClu_affinity, AggloClu_linkage):
        if linkage == 'ward' and affinity !='euclidean': continue
        print()
        print('############################################################################################')
        cluster_cfg = 'AggloClu with n_clusters: %s affinity: %s linkage: %s' % (n_clusters, affinity, linkage)
        print(vector_cfg)
        print(cluster_cfg)
        try:
            ac = AgglomerativeClustering(n_clusters=n_clusters, affinity =affinity, linkage=linkage)
            ac.fit(X.todense())
            get_results_with_n(ac.labels_, n_clusters, X, AggloClu_metrics, vector_cfg, cluster_cfg)
        except (ValueError, IndexError) as error:
            print("Caught error: %s" %error)
            continue

###KMeans parameters and definitoon####
KMeansClu_n_clusters = [7,8,9]
KMeansClu_max_iter = [100, 300, 900]
KMeansClu_init = ['k-means++', 'random']
KMeansClu_algorithm = ['full', 'elkan']
KMeansClu_tol =  [1e-4, 1e-3, 1e-2, 1e-1]
def run_KMeansClu(X, vector_cfg):
###run clustering
    for n_clusters, max_iter, init, algorithm, tol in product(KMeansClu_n_clusters, KMeansClu_max_iter, KMeansClu_init, KMeansClu_algorithm, KMeansClu_tol):
        #if linkage == 'ward' and affinity !='euclidean': continue
        print()
        print('############################################################################################')
        cluster_cfg = 'KMeans with n_clusters: %s max_iter: %s init: %s algorithm: %s tol: %s' % (n_clusters, max_iter, init, algorithm, tol)
        print(vector_cfg)
        print(cluster_cfg)
        try:
            ac = KMeans(n_clusters=n_clusters, max_iter=max_iter, init=init, algorithm=algorithm, tol=tol, precompute_distances = True, n_init=10, n_jobs = -1)
            ac.fit(X.todense())
            get_results_with_n(ac.labels_, n_clusters, X, KMeans_metrics, vector_cfg, cluster_cfg)
        except (ValueError, IndexError) as error:
            print("Caught error: %s" %error)
            continue
###Birch parameters and definitoon####
Birch_n_clusters = [7,8,9]
Birch_threshold = [0.1, 0.5 , 1.0]
Birch_branching_factor = [5, 20, 50]
def run_Birch(X, vector_cfg):
###run clustering
    for n_clusters, threshold, branching_factor in product(Birch_n_clusters, Birch_threshold, Birch_branching_factor):
        #if linkage == 'ward' and affinity !='euclidean': continue
        #if threshold != 0.5 or branching_factor !=50 or  n_clusters !=7: continue
        print()
        print('############################################################################################')
        cluster_cfg = 'Birch with n_clusters: %s threshold: %s branching_factor:%s' % (n_clusters, threshold, branching_factor)
        print(vector_cfg)
        print(cluster_cfg)
        try:
            ac = Birch(n_clusters=n_clusters, threshold=threshold, branching_factor=branching_factor)
            ac.fit(X.todense())
            get_results_with_n(ac.labels_, n_clusters, X, Birch_metrics, vector_cfg, cluster_cfg)
        except (ValueError, IndexError) as error:
            print("Caught error: %s" %error)
            continue
###Birch MeanShift and definitoon####
MeanShift_bin_seeding = [True, False]
MeanShift_cluster_all = [True, False]
def run_MeanShift(X, vector_cfg):
###run clustering
    for bin_seeding, cluster_all in product(MeanShift_bin_seeding, MeanShift_cluster_all):
        #if linkage == 'ward' and affinity !='euclidean': continue
        #if threshold != 0.5 or branching_factor !=50 or  n_clusters !=7: continue
        print()
        print('############################################################################################')
        cluster_cfg = 'MeanShift with bin_seeding: %s cluster_all: %s' % ( bin_seeding, cluster_all)
        print(vector_cfg)
        print(cluster_cfg)
        try:
            ac = MeanShift(bin_seeding = bin_seeding, cluster_all = bin_seeding)
            ac.fit(X.todense())
            get_results_without_n(ac.labels_, len(np.unique(ac.labels_)), X, MeanShift_metrics, vector_cfg, cluster_cfg)
        except (ValueError, IndexError) as error:
            print("Caught error: %s" %error)
            continue

AffinityPropagation_damping = [0.5, 0.75, 0.99]
AffinityPropagation_convergence_iter = [5 , 15 , 30]
AffinityPropagation_max_iter = [100, 200, 500]
AffinityPropagation_affinity = ['precomputed', 'euclidean']
def run_AffinityPropagation(X, vector_cfg):
###run clustering
    for damping, convergence_iter, max_iter, affinity in product(AffinityPropagation_damping, AffinityPropagation_convergence_iter, AffinityPropagation_max_iter, AffinityPropagation_affinity):
        #if threshold != 0.5 or branching_factor !=50 or  n_clusters !=7: continue
        print()
        print('############################################################################################')
        cluster_cfg = 'AffinityPropagation with damping: %s convergence_iter: %s max_iter: %s affinity: %s' % ( damping, convergence_iter, max_iter, affinity)
        print(vector_cfg)
        print(cluster_cfg)
        try:
            ac = AffinityPropagation(damping=damping, convergence_iter=convergence_iter, max_iter=max_iter, affinity=affinity)
            ac.fit(X.todense())
            get_results_without_n(ac.labels_, len(np.unique(ac.labels_)), X, AffinityPropagation_metrics, vector_cfg, cluster_cfg)
        except (ValueError, IndexError) as error:
            print("Caught error: %s" %error)
            continue

####TfidfVectorizer parameters####
TfidfVec_max_min = [[0.5,0.1],[0.8,0.1],[1.0,0.1],[0.8,0.5],[1.0,0.5],[1.0,0.8],[0.5,1],[0.8,1],
                    [1.0,1],[100,1],[0.5,2],[0.8,2],[1.0,2],[500,2],[0.5,20],[0.8,20],[1.0,20],[1000,20],[50,10] , [10,1], [0.8,100]]
TfidfVec_analyzer = ['word','char']
#TfidfVec_stop_words = ['english', None]
TfidfVec_lowercase = [True, False]
#TfidfVec_binary = [True, False]
TfidfVec_norm = ['l1', 'l2', None]
#TfidfVec_use_idf = [True, False]
#TfidfVec_smooth_idf = [True, False]
TfidfVec_sublinear_tf = [True, False]
#run TfidfVectorizer - 504 combinations
for analyzer, lowercase, norm, sublinear_tf, max_min in product(TfidfVec_analyzer, TfidfVec_lowercase, TfidfVec_norm, TfidfVec_sublinear_tf, TfidfVec_max_min):
    #if max_df != 0.4 or min_df != 2  or  norm !='l1' or  analyzer != 'word' or  use_idf != True or smooth_idf != True or sublinear_tf != False or binary!=False or lowercase!=True or stop_words != 'english':      continue
    print()
    print('%%%%%%%%%%%%%%%%%%Starting new TfidfVe cycle%%%%%%%%%%%%%%%%%%%%%%%%')
    vector_cfg = 'TfidfVe max_df: %s min_df: %s analyzer: %s lowercase: %s norm: %s sublinear_tf:%s' % (max_min[0], max_min[1], analyzer, lowercase, norm, sublinear_tf)
    try:
        vectorizer = TfidfVectorizer(max_df=max_min[0], min_df=max_min[1], analyzer=analyzer, stop_words='english', lowercase=lowercase, norm=norm, sublinear_tf=sublinear_tf)
        X = vectorizer.fit_transform(books_texts)
        print("n_samples: %d, n_features: %d" % X.shape)
        ###clusering which takes in predifined clusters
        run_AggloClu(X, vector_cfg)
        run_KMeansClu(X, vector_cfg)
        run_Birch(X, vector_cfg)
        ###clusering which does not take in predifined clusters
        run_MeanShift(X, vector_cfg)
        run_AffinityPropagation(X, vector_cfg)
    except ValueError as error:
        print("Caught error: %s" %error)
        continue

print('##############################################################################')
print('#################  Just finished TfidVec  ####################################')
print('###############################################################################')
print_res()



####CountVectorizer parameters####
CountVec_max_min = [[0.5,0.1],[0.8,0.1],[1.0,0.1],[0.8,0.5],[1.0,0.5],[1.0,0.8],[0.5,1],[0.8,1],
                    [1.0,1],[100,1],[0.5,2],[0.8,2],[1.0,2],[500,2],[0.5,20],[0.8,20],[1.0,20],[1000,20],[50,10] , [10,1], [0.8,100]]
CountVec_analyzer = ['word','char', 'char_wb']
#CountVec_stop_words = ['english', None]
CountVec_lowercase = [True, False]
#CountVec_binary = [True, False]

##run CountVectorizer - 126 combinations
for analyzer, lowercase, max_min in product(CountVec_analyzer, CountVec_lowercase, CountVec_max_min):
    print()
    print('%%%%%%%%%%%%%%%%%%Starting new CountVec cycle%%%%%%%%%%%%%%%%%%%%%%%%')
    vector_cfg = 'CountVec max_df: %s min_df: %s analyzer: %s lowercase: %s ' % (max_min[0], max_min[1], analyzer, lowercase)
    try:
        vectorizer = CountVectorizer(max_df=max_min[0], min_df=max_min[1], analyzer=analyzer, stop_words='english', lowercase=lowercase)
        X = vectorizer.fit_transform(books_texts)
        print("n_samples: %d, n_features: %d" % X.shape)
        ###clusering which takes in predifined clusters
        run_AggloClu(X, vector_cfg)
        run_KMeansClu(X, vector_cfg)
        run_Birch(X, vector_cfg)
        ###clusering which does not take in predifined clusters
        run_MeanShift(X, vector_cfg)
        run_AffinityPropagation(X, vector_cfg)
    except ValueError as error:
        print("Caught error: %s" %error)
        continue

print('#############################################################################')
print('#################  Just finished CountVectorizer ###############################')
print('##############################################################################')
print_res()

####HshingVectorizer parameters####
HashingVec_analyzer = ['word','char', 'char_wb']
HashingVec_n_features = [100000, 40000]
#HashingVec_stop_words = ['english', None]
HashingVec_lowercase = [True, False]
#HashingVec_binary = [True, False]
HashingVec_norm = ['l1', 'l2', None]
#HashingVec_non_negative = [True, False]
##run HashingVectorizer ##total 36 comibantions
for analyzer, n_features, lowercase, norm in product(HashingVec_analyzer, HashingVec_n_features,  HashingVec_lowercase, HashingVec_norm):
    print()
    print('%%%%%%%%%%%%%%%%%%Starting new HashVec cycle%%%%%%%%%%%%%%%%%%%%%%%%')
    vector_cfg = 'HashVec analyzer: %s n_features: %s lowercase: %s norm: %s' % (analyzer, n_features, lowercase, norm)
    try:
        vectorizer = HashingVectorizer(analyzer=analyzer, n_features=n_features, stop_words='english', lowercase=lowercase, norm=norm)
        X = vectorizer.fit_transform(books_texts)
        print("n_samples: %d, n_features: %d" % X.shape)
        ###clusering which takes in predifined clusters
        run_AggloClu(X, vector_cfg)
        run_KMeansClu(X, vector_cfg)
        run_Birch(X, vector_cfg)
        ###clusering which does not take in predifined clusters
        run_MeanShift(X, vector_cfg)
        run_AffinityPropagation(X, vector_cfg)
    except ValueError as error:
        print("Caught error: %s" %error)
        continue

print('#############################################################################')
print('#################  Just finished all ###############################')
print('##############################################################################')
print_res()
