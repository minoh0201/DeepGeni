import os
import time
import numpy as np
import pandas as pd
import pickle

import config


def load_OTUprofile_and_labels(OTU_profile_filename=config.OTUS, labels_filename=config.LABELS):
    # Time stamp
    start_time = time.time()
    cache_filename = os.path.join(os.getcwd(), 'data', OTU_profile_filename + "-" + labels_filename + ".pkl")
    
    # If not cached, cache data
    if not os.path.exists(cache_filename):

        # Filenames
        OTU_profile_filepath = os.path.join(os.getcwd(), 'data', OTU_profile_filename)
        labels_filepath = os.path.join(os.getcwd(), 'data', labels_filename)

        # Read OTU profile
        if os.path.isfile(OTU_profile_filepath):
            OTUs = pd.read_csv(OTU_profile_filepath, sep=',', index_col=0)
        else:  
            print(f"FileNotFoundError: {OTU_profile_filepath} does not exist")
            exit()

        # Read labels
        if os.path.isfile(labels_filepath):
            labels = pd.read_csv(labels_filepath, sep=',', index_col=0, header=None)
            labels = (labels == "R")
        else:  
            print(f"FileNotFoundError: {labels_filepath} does not exist")
            exit()

    # Return X, y, patient IDs, and feature names
    return OTUs.T, labels, OTUs.columns, OTUs.T.columns

def split_studies(X, y, ids):
    
    Xs = []
    ys = []

    # Get unique study indicator keeping order
    study_indicators = np.array([x.split('_')[1] for x in ids])
    _, idx = np.unique(study_indicators, return_index=True)
    study_indicators = study_indicators[np.sort(idx)]

    for study_indicator in study_indicators:
        study_str = "_" + study_indicator + "_"
        Xs.append(X[X.index.str.contains(study_str, regex=False)].values.astype(np.float))
        ys.append(y[X.index.str.contains(study_str, regex=False)].values.astype(np.int).flatten())

    return Xs, ys, study_indicators

        
        