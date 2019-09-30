#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 21:22:08 2019
Visualizing interesting trends in dating-full.csv 2-1
@author: liang257
"""

import pandas as pd
import sys
import matplotlib.pyplot as plt

'''import data from the command line'''
data = pd.read_csv(sys.argv[1])

'''
Problem (i)
'''
preference_scores_of_participant = ["attractive_important", "sincere_important", "intelligence_important",
"funny_important", "ambition_important", "shared_interests_important"]

#a. Divide the dataset into two sub-datasets by the gender of participant
dating_gender = data.groupby("gender")

#Within each sub-dataset, compute the mean values for each column in the set preference scores of participant
means_scores_participant_gender = dating_gender[preference_scores_of_participant].mean()

#Use a single barplot to contrast how females and males value the six attributes in their romantic partners differently.
means_scores_participant_gender.T.plot(kind="bar")
plt.xlabel("Attributes")
plt.ylabel("Mean values of preference scores of participant")
plt.xticks(rotation=10)

plt.savefig("Ass2_2_1.png")