#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 12:53:27 2019
Training-Test Split
@author: liang257
"""

import sys
import pandas as pd

'''read files from command lines'''
in_file = sys.argv[1]
out_file1 = sys.argv[2]
out_file2 = sys.argv[3]

data = pd.read_csv(in_file)

'''split train and test set'''
frac = 0.2
random_state = 47

test_data = data.sample(frac=frac, replace=False, random_state=random_state)
train_data = data.drop(test_data.index)

'''save the train and test set'''
train_data.to_csv(out_file1, index=False)
test_data.to_csv(out_file2, index=False)