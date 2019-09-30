#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 22:08:17 2019
Visualizing interesting trends in dating-full.csv 2-2
@author: liang257
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(sys.argv[1])

rating_of_partner_from_participant = ["attractive_partner", "sincere_partner", "intelligence_parter", "funny_partner", "ambition_partner", "shared_interests_partner"]

#determine the number of distinct values for attractive_partner.
uniq_num = data["attractive_partner"].nunique()

#compute the success rate for attractive_partner at level 10
num_attr_part = len(data.loc[data["attractive_partner"]==10])
des_attr_part = len(data.loc[(data["attractive_partner"]==10) & (data["decision"]==1)])
success_rate_attr_part = des_attr_part / num_attr_part

#repeat the above process for all distinct values on each of the six attributes in the set rating of partner from participant.
success_rate = []

for col in rating_of_partner_from_participant:
    df = pd.DataFrame(columns=["Attributes", "Values", "Success_Rate"])
    df["Values"] = np.sort(data[col].unique())
    for i, value in enumerate(df["Values"]):
        df.iloc[i,0] = col
        num = len(data.loc[data[col] == value])
        des= len(data.loc[(data[col] == value) & (data["decision"] == 1)])
        df.iloc[i, 2] = des / num
    success_rate.append(df)

success_rate = pd.concat(success_rate, axis=0)

#draw a scatter plot for each of the 6 attributes
for attr in rating_of_partner_from_participant:
    df = success_rate.loc[success_rate["Attributes"]==attr]
    plt.figure()
    plt.scatter("Values", "Success_Rate", data = df)
    plt.xlabel(col)
    plt.ylabel("Values")
    plt.savefig("Ass2_2_2_"+attr+".png")

