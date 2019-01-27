import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

features = {}

def load_data_fm(train_file,test_file,interval):
    count_num_feature_field(train_file,interval)
    count_num_feature_field(test_file,interval)
    features_M = len(features)

    train_data = read_data(train_file,interval)
    test_data = read_data(test_file,interval)

    return train_data, test_data, features_M


def count_num_feature_field(file,interval):
    f = open(file)
    line = f.readline()
    i = len(features)
    while line:
         elements = line.strip().split(interval)
         for e in elements[1:]:
             if e not in features:
                 features[e] = i
                 i = i + 1
         line = f.readline()
    f.close()

def read_data(file,interval):
    f = open(file)
    X = []
    Y = []

    line = f.readline()
    while line:
        elements = line.strip().split(interval)
        Y.append([float(elements[0])])
        X.append([ features[e] for e in elements[1:]])
        line = f.readline()
    f.close()
    Data_Dict = {}
    Data_Dict['Y'] = Y
    Data_Dict['X'] = X
    return Data_Dict










