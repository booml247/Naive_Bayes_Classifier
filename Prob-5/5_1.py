#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 13:13:47 2019
Implement a Naive Bayes Classifier
@author: liang257
"""

import pandas as pd
import numpy as np

'''read data'''
train_data = pd.read_csv("trainingSet.csv")
test_data = pd.read_csv("testSet.csv")

'''helper function'''
def count(data, attr, attr_spec, dec_spec):
    return len(data.loc[(data[attr] == attr_spec) & (data["decision"] == dec_spec)])



'''naive bayes classifier function'''
def nbc(t_frac):
    #split the data
    random_state = 47
    train = train_data.sample(frac=t_frac, replace=False, random_state= random_state)

    #obtain the prior
    train_n = len(train)

    count_0 = count(train, "decision", 0, 0)
    count_1 = count(train, "decision", 1, 1)

    p_0 = count_0 / train_n
    p_1 =  count_1 / train_n

    #obtain the CPDs
    cpd = {0: {}, 1: {}}

    for col in train.columns[:-1]:
        cpd[0][col] = pd.Series([])
        cpd[1][col] = pd.Series([])

        # apply laplace correction
        if col in continuous_valued_columns_spec:
            for val in range(bin_nums):
                cpd[0][col].loc[val] = (count(train, col, val, 0) + 1) / (count_0 + bin_nums)
                cpd[1][col].loc[val] = (count(train, col, val, 1) + 1) / (count_1 + bin_nums)
                #cpd[0][col].loc[val] = (count(train, col, val, 0)) / (count_0)
                #cpd[1][col].loc[val] = (count(train, col, val, 1)) / (count_1)
        elif col == "race" or col == "race_o":
            for val in range(5):
                #cpd[0][col].loc[val] = (count(train, col, val, 0) + 1) / (count_0 + 5)
                #cpd[1][col].loc[val] = (count(train, col, val, 1) + 1) / (count_1 + 5)
                cpd[0][col].loc[val] = (count(train, col, val, 0)) / (count_0)
                cpd[1][col].loc[val] = (count(train, col, val, 1)) / (count_1)
        elif col == "gender" or col == "samerace":
            for val in range(2):
                cpd[0][col].loc[val] = (count(train, col, val, 0) + 1) / (count_0 + 2)
                cpd[1][col].loc[val] = (count(train, col, val, 1) + 1) / (count_1 + 2)
                #cpd[0][col].loc[val] = (count(train, col, val, 0)) / (count_0)
                #cpd[1][col].loc[val] = (count(train, col, val, 1)) / (count_1)
        elif col == "field":
            for val in range(210):
                cpd[0][col].loc[val] = (count(train, col, val, 0) + 1) / (count_0 + 210)
                cpd[1][col].loc[val] = (count(train, col, val, 1) + 1) / (count_1 + 210)
                #cpd[0][col].loc[val] = (count(train, col, val, 0)) / (count_0)
                #cpd[1][col].loc[val] = (count(train, col, val, 1)) / (count_1)
    return p_0, p_1, cpd


'''implement nbc on the train and test set'''
#obtain the estimate of priors and cpds
spec_columns_spec = ["gender", "race", "race_o", "samerace", "field"]
continuous_valued_columns_spec = [i for i in train_data.columns if i not in spec_columns_spec]
bin_nums = 5

p_0, p_1, cpd = nbc(1)

#make predictions
n = len(train_data)
test_n = len(test_data)
pred_train = []
pred_test = []

for i in range(n):
    pred_0 = p_0
    pred_1 = p_1

    for col in train_data.columns[:-1]:
        pred_0 *= cpd[0][col].loc[train_data[col].iloc[i]]
        pred_1 *= cpd[1][col].loc[train_data[col].iloc[i]]

    pred_train.append(int(pred_1 > pred_0))

print("Train Accuracy: ", round(sum(pred_train == train_data["decision"])/n, 2))

for i in range(test_n):
    pred_0 = p_0
    pred_1 = p_1

    for col in test_data.columns[:-1]:
        pred_0 *= cpd[0][col].loc[test_data[col].iloc[i]]
        pred_1 *= cpd[1][col].loc[test_data[col].iloc[i]]

    pred_test.append(int(pred_1 > pred_0))

print("Test Accuracy: ", round(sum(pred_test == test_data["decision"])/test_n, 2))




