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
method = 'sketch'
times = 10
base_training_num = (datetime.strptime('2020-06-01', '%Y-%m-%d')-datetime.strptime('2020-03-01', '%Y-%m-%d')).days
base_testing_num = 30
delta_day = 14


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)


    return np.mean(np.abs((y_true - y_pred) / y_true))



for i_country, country in enumerate(countries):

    
    input_file = country +'_data.txt'
    groundtruth_file = country + '_impact.mat'
    
    mses = np.zeros((len(hash_nums),  times))
    mapes = np.zeros((len(hash_nums), times))
    elapses = np.zeros((len(hash_nums), times))
    for i_hash_num, hash_num in enumerate(hash_nums):
    
        f = open(input_file)
        line = f.readline()
        date_policy_dict = {}
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
            line = f.readline()
            pair_id = pair_id+1
            if date < datetime.strptime('2020-06-01', '%Y-%m-%d'):
                i_training = i_training+1
        f.close()
        

        predicted_labels_list = []
        for i_times in range(times):
            feature_id_dict = {}
            vectorizer = CountVectorizer()
            
            x_random_list = []
            idf_list = []
            fingerprints = []
            policy_id=1
            
            def get_feature_id(feature):
                if feature not in feature_id_dict.keys():
                    feature_id_dict[feature] = len(feature_id_dict)
                    x_random_list.append(np.random.uniform(0,1, (1, hash_num)))
                return feature_id_dict[feature]
            
            start = time.time()    
            
            X = vectorizer.fit_transform(date_policy_dict[starting_date])
            features = vectorizer.get_feature_names()
            ids = np.array([get_feature_id(feature) for feature in features])
            idf_list = [1]*len(ids)
            weights = np.transpose(X.toarray().astype(float)/len(date_policy_dict[starting_date][0].split(' ')))
            
            
            
            x_random = np.concatenate(x_random_list)[ids]
            hash_values = np.divide(-np.log(x_random), weights)
            min_ids = ids[np.argmin(hash_values, axis=0)]
            fingerprints.append(min_ids)
            
            policy_id = policy_id+1
            if i_country == 0:
                window_size = 4
            elif i_country == 1:
                window_size = 1
            else:
                window_size = 3
            for i_date, date in enumerate(sorted(date_policy_dict.keys())):
                if i_date==0:
                    continue
                
                # adjust hash values
                coefficients = np.divide(np.log(np.divide(float(policy_id+1), np.array(idf_list))), np.log(np.divide(float(policy_id), np.array(idf_list)))).reshape(-1,1)

                # amplify
                weights = np.multiply(weights, coefficients)
                hash_values = np.multiply(hash_values, coefficients)
                hash_value_list = list(hash_values)
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
                            x_random = np.array(x_random_list[id])
                            hash_value_list[id] = np.divide(-np.log(x_random), weight_list[id])[0,:]
                        else:
                            weight_list.append(weights2[i_id,0])
                            idf_list.append(1)
                            x_random = np.array(x_random_list[id])
                            hash_value_list.append(np.divide(-np.log(x_random), weight_list[id])[0,:])
                    hash_values = np.array(hash_value_list)
                    weights = np.array(weight_list)
                    weights = MinMaxScaler().fit_transform(weights.reshape(-1,1))
                    
                ids = np.arange(len(feature_id_dict))   
                min_ids = ids[np.argmin(hash_values, axis=0)]   
                fingerprints.append(min_ids)
                policy_id = policy_id+1   
                    
            fingerprints = np.array(fingerprints)
            labels = sio.loadmat(groundtruth_file)['impact']
            predicted_labels = []
            
            testing_labels = labels[base_training_num:,0]
        
            
            for i_testing in range(0, base_testing_num):
                training_labels = labels[base_training_num+i_testing-window_size:base_training_num+i_testing,0]  
                
                training_set = fingerprints[base_training_num+i_testing-delta_day-window_size : base_training_num+i_testing-delta_day,:]
                testing_set = fingerprints[base_training_num+i_testing-delta_day:base_training_num+i_testing-delta_day+1,:]
                
                training_kernel = 1-distance.cdist(training_set, training_set, lambda u, v: np.divide(np.sum(np.minimum(u, v)), np.sum(np.maximum(u, v)), out=np.zeros_like(np.sum(np.minimum(u, v))), where=np.sum(np.maximum(u, v))!=0))
                
                regr = SVR(kernel='precomputed')

                regr.fit(training_kernel, training_labels)
                testing_kernel = 1-distance.cdist(testing_set, training_set, lambda u, v: np.divide(np.sum(np.minimum(u, v)), np.sum(np.maximum(u, v)), out=np.zeros_like(np.sum(np.minimum(u, v))), where=np.sum(np.maximum(u, v))!=0))
                predicted_labels.append(regr.predict(testing_kernel)[0])
            
            elapses[i_hash_num,  i_times] = time.time() - start
            predicted_labels=np.array(predicted_labels)
            mapes[i_hash_num,  i_times] = mean_absolute_percentage_error(testing_labels, predicted_labels)
            predicted_labels_list.append(predicted_labels)
            predicted_labels = np.array(predicted_labels_list)
            
            sio.savemat(country+ '_results.mat', {'fingerprints':fingerprints, 'mapes': mapes,'testing_labels':testing_labels, 'predicted_labels': predicted_labels})        
