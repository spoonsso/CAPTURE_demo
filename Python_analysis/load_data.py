'''
load_data.py

Functions for reading in parameters and data and serving appropriate structures

Joshua Wu
29 March, 2022
'''
import numpy as np

def read_config(filename):
    '''
    Read configuration file.

    IN:
        filename - Path to configuration file.
    OUT:
        config_params - Dict of params in config file
    '''
    import yaml
    with open(filename) as f:
        config_params = yaml.safe_load(f)
    return config_params


def load_data(analysis_path, preds_path, batch_name, subsample=30):
    '''
    Load in data (we only care about batchID, frames_with_good_tracking and jt_features)

    IN:
        analysis_path - path to MATLAB analysis struct with jt_features included
        preds_path - path to predictions .mat file
        batch_name - name of category to split for batch maps
        subsample - factor by which to downsample features and IDs for analysis

    OUT:
        features - Numpy array of features for each frame for analysis (frames x features)
        batch_ID - List of labels for categories based on the batch_name
    '''
    import hdf5storage
    analysisstruct = hdf5storage.loadmat(analysis_path, variable_names=['jt_features','frames_with_good_tracking'])
    batch_IDs_full = np.squeeze(hdf5storage.loadmat(preds_path, variable_names=[batch_name])[batch_name].astype(int))
    features = analysisstruct['jt_features']
    frames_with_good_tracking = np.squeeze(analysisstruct['frames_with_good_tracking'][0][1].astype(int))

    batch_ID = batch_IDs_full[frames_with_good_tracking] # Indexing out batch IDs

    # Subsample by 30
    features = features[::subsample]
    batch_ID = batch_ID[::subsample]

    return [features, batch_ID]

