import os

import pandas as pd

from featurizer import Featurizer
from classifier import Classifier


RERUN_FEATURES = False


def main():
    data_dir = (os.path.join(os.getcwd(), "Assignments", "Assignment 4", "data"))
    
    featurizer = Featurizer()

    full_train_set = get_features(featurizer, RERUN_FEATURES, feature_filename=os.path.join(data_dir, "train_features.csv"), filename=os.path.join(data_dir, "pan2324_train_data.csv"))
    full_dev_set = get_features(featurizer, RERUN_FEATURES, feature_filename=os.path.join(data_dir, "dev_features.csv"), filename=os.path.join(data_dir, "pan2324_dev_data.csv"))
    # full_test_set = get_features(featurizer, RERUN_FEATURES, feature_filename=os.path.join(data_dir, "test_features.csv"), filename=os.path.join(data_dir, "pan2324_test_data.csv"))

    print("Running classifier...")
    classifier = Classifier()
    train_x = full_train_set[[col for col in full_train_set.columns if col.startswith("f_")]].to_numpy()
    train_y = full_train_set["author"].to_numpy()
    classifier.fit(train_x, train_y)

    print("Evaluating classifier...")
    dev_x = full_dev_set[[col for col in full_train_set.columns if col.startswith("f_")]].to_numpy()
    dev_y = full_dev_set["author"].to_numpy()
    classifier.evaluate(dev_x, dev_y)


def get_features(featurizer: Featurizer, rerun_features, feature_filename, filename):   
    data_set = pd.read_csv(filename, index_col=0)
    data_set.index.names = ["index"]
    
    if rerun_features or not os.path.exists(feature_filename):
        print("Calculating features...")
        features = featurizer.featurize(data_set)
        features.to_csv(feature_filename)
        summary = features.describe(include='all')
        summary.to_csv(f"{feature_filename.split('.')[0].split('_')[0]}_summary.csv")
    else:
        print("Loading features...")
        features = pd.read_csv(feature_filename, index_col="index")

    return pd.merge(data_set, features, on="index", how="inner")
    

if __name__ == "__main__":
    main()
