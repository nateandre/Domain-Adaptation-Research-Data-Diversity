""" Helper functions, including for loading data and creating features.

Author: Nathaniel Andre
"""

import numpy as np


def get_labels_and_features(nested_embeddings):
    """ returns labels and embeddings
    """
    x = nested_embeddings[:,:-1]
    y = nested_embeddings[:,-1]
    return x,y


def load_data(target_dir,source_dirs,partition_n,target_train_size,source_cutoff=20000,use_source_cutoff=False,data_dir="../data/"):
    """ Loads the target and source datasets
    args:
        target_dir: name of dir of the target-domain data
        source_dirs: name(s) of dirs for source-domain (can be empty)
        source_cutoff: index if wanted to use a subset of the source-domain data
        use_source_cutoff (bool): whether to use a subset of source-domain data
    """
    target_train = np.load(data_dir+target_dir+"/target_train_n{}_p{}.npy".format(target_train_size,partition_n))
    target_val = np.load(data_dir+target_dir+"/target_val_p{}.npy".format(partition_n))
    target_test = np.load(data_dir+target_dir+"/target_test_p{}.npy".format(partition_n))
    source_trains = [np.load(data_dir+source_dir+"/source_train_p{}.npy".format(partition_n)) for source_dir in source_dirs]

    target_train_x,target_train_y = get_labels_and_features(target_train)
    target_val_x,target_val_y = get_labels_and_features(target_val)
    target_test_x,target_test_y = get_labels_and_features(target_test)
    source_data = [get_labels_and_features(source_train) for source_train in source_trains]    

    if use_source_cutoff:
        source_data = [(pair[0][:source_cutoff],pair[1][:source_cutoff]) for pair in source_data]

    source_training_data = [] # flattening the source_data
    for pair in source_data:
        source_training_data.append(pair[0])
        source_training_data.append(pair[1])
    
    return target_train_x,target_train_y,target_val_x,target_val_y,target_test_x,target_test_y,source_training_data


def get_mean(lis):
    """ returns the mean of the items in a list
    """
    return sum(lis)/len(lis)
