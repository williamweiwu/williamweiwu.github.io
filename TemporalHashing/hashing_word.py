#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:23:18 2020

@author: weiwu
"""
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np
import numpy.matlib
import time
import scipy as sp
import scipy.io as sio
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

countries = ['Australia','America', 'UK']
hash_nums = [100]
method = 'hashing_word'
times = 1
base_training_num = (datetime.strptime('2020-06-01', '%Y-%m-%d')-datetime.strptime('2020-03-01', '%Y-%m-%d')).days
base_testing_num = 30
delta_day = 14



country_word_sketch_dict = {}
country_id_sketch_dict = {}
country_keyword_sketch_dict = {}
country_policy_dict = {}

for i_country, country in enumerate(countries):
    
    input_file = country +'_data.txt'
    groundtruth_file = country + '_impact.mat'
    
    
    
    f = open(input_file)
    line = f.readline()
    date_policy_dict = {}
    policy_dict = {}
    pair_id = 0
    i_training = 0
    while line:
        line = line.strip()
        date = datetime.strptime(line.split('\t')[0], '%Y-%m-%d')
        if date >= datetime.strptime('2020-07-01', '%Y-%m-%d') or date < datetime.strptime('2020-03-01', '%Y-%m-%d'):
            line = f.readline()
            continue
        if pair_id ==0:
            starting_date = date
        if len(line.split('\t')) == 2:
            	policy = [line.split('\t')[1]]
        else:
            	policy = []
        date_policy_dict[date] = policy
        policy_dict[pair_id] = policy
        line = f.readline()
        pair_id = pair_id+1
        if date < datetime.strptime('2020-06-01', '%Y-%m-%d'):
            i_training = i_training+1
    f.close()
        

    feature_id_dict = {}
    id_feature_dict = {}
    response_weight_list = []
    vectorizer = CountVectorizer()
    
    idf_list = []
    fingerprints = []
    policy_id=1
    
    def get_feature_id(feature):
        if feature not in feature_id_dict.keys():
            feature_id_dict[feature] = len(feature_id_dict)
            id_feature_dict[len(feature_id_dict)-1] = feature
        return feature_id_dict[feature]
    
    
    X = vectorizer.fit_transform(date_policy_dict[starting_date])
    features = vectorizer.get_feature_names()
    ids = np.array([get_feature_id(feature) for feature in features])
    idf_list = [1]*len(ids)
    if i_country == 0:
        window_size = 4
        weight_time = np.zeros((9163,122))
    elif i_country == 1:
        window_size = 1
        weight_time = np.zeros((11857,122))
    else:
        window_size = 3
        weight_time = np.zeros((12888,122))
    weights = np.transpose(X.toarray().astype(float)/len(date_policy_dict[starting_date][0].split(' ')))
    weights = weights.reshape(-1,1)
    weight_time[0:weights.shape[0], policy_id-1] = weights[:,0] 
    
    
    policy_id = policy_id+1
        
    for i_date, date in enumerate(sorted(date_policy_dict.keys())):
        if i_date==0:
            continue
        
        # adjust hash values
        coefficients = np.divide(np.log(np.divide(float(policy_id+1), np.array(idf_list))), np.log(np.divide(float(policy_id), np.array(idf_list)))).reshape(-1,1)

        # amplify
        weights = np.multiply(weights, coefficients)
        weight_list = list(weights[:,0])
        
        if date_policy_dict[date]:
            
            X = vectorizer.fit_transform(date_policy_dict[date])
            features = vectorizer.get_feature_names()
            ids = np.array([get_feature_id(feature) for feature in features])
            weights2 = np.transpose(X.toarray().astype(float)/len(date_policy_dict[date][0].split(' ')))
            for i_id, id in enumerate(ids):
                if id < weights.shape[0]:
                    weight_list[id] = weights[id,0]+weights2[i_id,0]
                    idf_list[id] = idf_list[id]+1
                else:
                    weight_list.append(weights2[i_id,0])
                    idf_list.append(1)
        weights = np.array(weight_list)
        weights = weights.reshape(-1,1)
        weight_time[0:weights.shape[0], policy_id-1] = weights[:,0]
            
        ids = np.arange(len(feature_id_dict))   
        policy_id = policy_id+1   
            

    country_word_sketch_dict[country] = weight_time
    country_id_sketch_dict[country] = feature_id_dict    
    
    
    keywords = ['lockdown', 'border']
    keyword_weights = np.zeros((len(keywords), 122))
    for i_keyword, keyword in enumerate(keywords):
        index = feature_id_dict[keyword]
        keyword_weights[i_keyword,:] = weight_time[index,:]
    country_keyword_sketch_dict[country] = keyword_weights
    country_policy_dict[country] = policy_dict
    sio.savemat(country+ '_'+method+'.mat', {'keywords':country_keyword_sketch_dict[country]})   

    del date_policy_dict
    del feature_id_dict
    del id_feature_dict
    
         

