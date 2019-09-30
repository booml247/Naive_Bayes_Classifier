#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 18:21:19 2019
Implement a Naive Bayes Classifier
@author: liang257
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


'''helper function'''
def count(data, attr, attr_spec, dec_spec):
    return len(data.loc[(data[attr] == attr_spec) & (data["decision"] == dec_spec)])



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

        # apply with/without laplace correction
        if col in continuous_valued_columns:
            for val in range(bin_nums):
                cpd[0][col].loc[val] = (count(train, col, val, 0) + 1) / (count_0 + bin_nums)
                cpd[1][col].loc[val] = (count(train, col, val, 1) + 1) / (count_1 + bin_nums)
                # cpd[0][col].loc[val] = (count(train, col, val, 0)) / (count_0)
                # cpd[1][col].loc[val] = (count(train, col, val, 1)) / (count_1)
        elif col == "race" or col == "race_o":
            for val in range(5):
                cpd[0][col].loc[val] = (count(train, col, val, 0) + 1) / (count_0 + 5)
                cpd[1][col].loc[val] = (count(train, col, val, 1) + 1) / (count_1 + 5)
                # cpd[0][col].loc[val] = (count(train, col, val, 0)) / (count_0)
                # cpd[1][col].loc[val] = (count(train, col, val, 1)) / (count_1)
        elif col == "gender" or col == "samerace":
            for val in range(2):
                cpd[0][col].loc[val] = (count(train, col, val, 0) + 1) / (count_0 + 2)
                cpd[1][col].loc[val] = (count(train, col, val, 1) + 1) / (count_1 + 2)
                # cpd[0][col].loc[val] = (count(train, col, val, 0)) / (count_0)
                # cpd[1][col].loc[val] = (count(train, col, val, 1)) / (count_1)
        elif col == "field":
            for val in range(210):
                cpd[0][col].loc[val] = (count(train, col, val, 0) + 1) / (count_0 + 210)
                cpd[1][col].loc[val] = (count(train, col, val, 1) + 1) / (count_1 + 210)
                # cpd[0][col].loc[val] = (count(train, col, val, 0)) / (count_0)
                # cpd[1][col].loc[val] = (count(train, col, val, 1)) / (count_1)

    return p_0, p_1, cpd

def discretize(data, bin_nums):
    for col in continuous_valued_columns:
        count = []
        # for cols in preference_scores_of_participant and preference_scores_of_partner
        if col in preference_scores_of_participant + preference_scores_of_partner:
            inter = np.linspace(0, 1, bin_nums + 1)
            inter[0] = inter[0] - e
            data[col] = pd.cut(data[col], bins=inter, right=True, labels=labels)
            for i in range(bin_nums):
                count.append(sum(data[col].cat.codes == i))
            #print(col + ": " + str(count))

        # for cols in age_cor
        elif col in age_cor:
            inter = np.linspace(18, 58, bin_nums + 1)
            inter[0] = inter[0] - e
            data[col] = pd.cut(data[col], bins=inter, right=True, labels=labels)
            for i in range(bin_nums):
                count.append(sum(data[col].cat.codes == i))
            #print(col + ": " + str(count))

        # for cols in interest_cor
        elif col in interest_cor:
            inter = np.linspace(-1, 1, bin_nums + 1)
            inter[0] = inter[0] - e
            data[col] = pd.cut(data[col], bins=inter, right=True, labels=labels)
            for i in range(bin_nums):
                count.append(sum(data[col].cat.codes == i))
            #print(col + ": " + str(count))

        # for other cols
        else:
            inter = np.linspace(0, 10, bin_nums + 1)
            inter[0] = inter[0] - e
            data[col] = pd.cut(data[col], bins=inter, right=True, labels=labels)
            for i in range(bin_nums):
                count.append(sum(data[col].cat.codes == i))
            #print(col + ": " + str(count))

    return data


'''Problem 5.3'''
F = [0.01, 0.1, 0.2, 0.5, 0.6, 0.75, 0.9, 1]
train_acc_path = []
test_acc_path = []
bin_nums = 5
labels = range(bin_nums)

for f in F:
    '''read data'''
    train_data = pd.read_csv("trainingSet.csv")
    test_data = pd.read_csv("testSet.csv")

    '''get the continuous valued columns'''
    cat_columns = ["gender", "race", "race_o", "samerace", "field", "decision"]
    continuous_valued_columns = [i for i in train_data.columns if i not in cat_columns]
    preference_scores_of_participant = ["attractive_important", "sincere_important", "intelligence_important",
                                        "funny_important", "ambition_important", "shared_interests_important"]
    preference_scores_of_partner = ["pref_o_attractive", "pref_o_sincere", "pref_o_intelligence", "pref_o_funny",
                                    "pref_o_ambitious", "pref_o_shared_interests"]
    age_cor = ["age", "age_o"]
    interest_cor = ["interests_correlate"]

    e = 1e-5

    '''apply nbc'''
    spec_columns = ["gender", "samerace", "field"]
    continuous_valued_columns_spec = [i for i in train_data.columns if i not in spec_columns]
    p_0, p_1, cpd = nbc(f)

    # make predictions
    # split the data
    random_state = 47
    train = train_data.sample(frac=f, replace=False, random_state=random_state)

    n = len(train)
    test_n = len(test_data)
    pred_train = []
    pred_test = []

    for i in range(n):
        pred_0 = p_0
        pred_1 = p_1

        for col in train_data.columns[:-1]:
            pred_0 *= cpd[0][col].loc[train[col].iloc[i]]
            pred_1 *= cpd[1][col].loc[train[col].iloc[i]]

        pred_train.append(int(pred_1 > pred_0))

    train_acc = round(sum(pred_train == train["decision"]) / n, 2)
    train_acc_path.append(train_acc)
    print("Train Accuracy: ", train_acc)

    for i in range(test_n):
        pred_0 = p_0
        pred_1 = p_1

        for col in test_data.columns[:-1]:
            pred_0 *= cpd[0][col].loc[test_data[col].iloc[i]]
            pred_1 *= cpd[1][col].loc[test_data[col].iloc[i]]

        pred_test.append(int(pred_1 > pred_0))

    test_acc = round(sum(pred_test == test_data["decision"]) / test_n, 2)
    test_acc_path.append(test_acc)
    print("Test Accuracy: ", test_acc)

# create plot
rects1 = plt.plot(F, train_acc_path, color='blue', linewidth=1.0, marker='o', label='Training')

rects2 = plt.plot(F, test_acc_path, color='green', linewidth=1.0, marker='o', label='Test')

plt.xlabel('fraction of the training dataset')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('Ass2_5_3.png')







