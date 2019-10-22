#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 09:40:17 2019
Convert continuous attributes to categorical attributes
@author: liang257
"""

import sys
import pandas as pd
import numpy as np

'''read files from the command line'''
in_file = sys.argv[1]
out_file = sys.argv[2]

data = pd.read_csv(in_file)
data["gaming"] = data["gaming"].mask(data["gaming"] > 10, 10)
data["reading"] = data["reading"].mask(data["reading"] > 10, 10)


'''get the continuous valued columns'''
cat_columns = ["gender", "race", "race_o", "samerace", "field", "decision"]
continuous_valued_columns = [i for i in data.columns if i not in cat_columns]
preference_scores_of_participant = ["attractive_important", "sincere_important", "intelligence_important",
"funny_important", "ambition_important", "shared_interests_important"]
preference_scores_of_partner = ["pref_o_attractive", "pref_o_sincere", "pref_o_intelligence", "pref_o_funny",
"pref_o_ambitious", "pref_o_shared_interests"]
age_cor = ["age", "age_o"]
interest_cor = ["interests_correlate"]


'''discretize the continous valued columns'''
bin_nums = 5
e = 1e-5
labels = range(bin_nums)

def discretize(data, bin_nums):
    for col in continuous_valued_columns:
        count = []
        # for cols in preference_scores_of_participant and preference_scores_of_partner
        if col in preference_scores_of_participant + preference_scores_of_partner:
            inter = np.linspace(0, 1, bin_nums + 1)
            inter[0] = inter[0] - e
            data[col] = pd.cut(data[col], bins=inter, right=True, labels=labels)
            for i in range(5):
                count.append(sum(data[col].cat.codes == i))
            print(col + ": " + str(count))

        # for cols in age_cor
        elif col in age_cor:
            inter = np.linspace(18, 58, bin_nums + 1)
            inter[0] = inter[0] - e
            data[col] = pd.cut(data[col], bins=inter, right=True, labels=labels)
            for i in range(5):
                count.append(sum(data[col].cat.codes == i))
            print(col + ": " + str(count))

        # for cols in interest_cor
        elif col in interest_cor:
            inter = np.linspace(-1, 1, bin_nums + 1)
            inter[0] = inter[0] - e
            data[col] = pd.cut(data[col], bins=inter, right=True, labels=labels)
            for i in range(5):
                count.append(sum(data[col].cat.codes == i))
            print(col + ": " + str(count))

        # for other cols
        else:
            inter = np.linspace(0, 10, bin_nums + 1)
            inter[0] = inter[0] - e
            data[col] = pd.cut(data[col], bins=inter, right=True, labels=labels)
            for i in range(5):
                count.append(sum(data[col].cat.codes == i))
            print(col + ": " + str(count))

    return data

data = discretize(data, bin_nums)

data.to_csv(out_file, index=False)
