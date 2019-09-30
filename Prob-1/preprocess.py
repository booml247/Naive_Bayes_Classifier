#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 16:48:58 2019
preprocess the data in dating-full.csv
@author: liang257
"""

import sys
import pandas as pd
import numpy as np

'''read arguments from the command line'''
in_file = sys.argv[1]
out_file = sys.argv[2]

'''read the csv file'''
data = pd.read_csv(in_file)

'''
Problem (i)
'''
quote_count = 0
quote_attri = ['race', 'race_o', 'field']
for col in quote_attri:
    quote_count += sum(data[col].str.count("'")/2)
    data[col] = data[col].str.replace("'","")
print("Quotes removed from ", quote_count, " cells.")

'''
Problem (ii)
'''
casechange_count = len(data['field'])-sum(data['field'].str.islower()) #count the number of cells that will be changed
data['field'] = data['field'].str.lower() #convert all the values in the column field to lowercase
print("Standardized ", casechange_count, " cells to lower case.")

'''
Problem (iii)
'''
#Output categorical value of the attributes
print("Value assigned for male in column gender: ", np.where(np.sort(data["gender"].unique())=="male")[0][0])
print("Value assigned for European/Caucasian-American in column race: ", np.where(np.sort(data["race"].unique())=="European/Caucasian-American")[0][0])
print("Value assigned for Latino/Hispanic American in column race_o: ", np.where(np.sort(data["race_o"].unique())=="Latino/Hispanic American")[0][0])
print("Value assigned for law in column field: ", np.where(np.sort(data["field"].unique())=="law")[0][0])

#convert the categorical values in columns gender, race, race o and field to numeric values start from 0
map_attri = ["gender", "race", "race_o", "field"]
for col in map_attri:
    uniq_list = np.sort(data[col].unique())
    cat = 0
    for word in uniq_list:
        data[col] = data[col].replace(word, cat)
        cat += 1

'''
Problem (iv)
'''
pso_participant = ["attractive_important", "sincere_important", "intelligence_important", "funny_important", "ambition_important", "shared_interests_important"]
pso_partner = ["pref_o_attractive", "pref_o_sincere", "pref_o_intelligence", "pref_o_funny", "pref_o_ambitious", "pref_o_shared_interests"]

total1 = data[pso_participant].sum(axis=1)
data[pso_participant] = data[pso_participant].div(total1, axis = 0)

total2 = data[pso_partner].sum(axis=1)
data[pso_partner] = data[pso_partner].div(total2, axis = 0)

#mean of each attribute
for col in pso_participant:
    print("Mean of ", col, ": ", round(data[col].mean(0),2))

for col in pso_partner:
    print("Mean of ", col, ": ", round(data[col].mean(0),2))

'''
Save the data to output file
'''
data.to_csv(out_file, index = False)