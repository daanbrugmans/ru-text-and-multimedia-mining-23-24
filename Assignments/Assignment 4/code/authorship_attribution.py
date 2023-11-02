import os

import pandas as pd

from featurizer import Featurizer
from classifier import Classifier


RERUN_FEATURES = False
os.chdir('[/home/janneke/Documents/Master/Text and Multimedia Mining/Assignment 4/code]')


def main():
    featurizer = Featurizer()

    full_train_set = get_features(featurizer, RERUN_FEATURES, feature_filename="../data/train_features.csv", filename="../data/pan2324_train_data.csv")
    full_dev_set = get_features(featurizer, RERUN_FEATURES, feature_filename="../data/dev_features.csv", filename="../data/pan2324_dev_data.csv")
    # full_test_set = get_features(featurizer, RERUN_FEATURES, feature_filename="../data/test_features.csv", filename="../data/pan2324_test_data.csv")

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
    if rerun_features or not os.path.exists(feature_filename):
        print("Calculating features...")
        features = featurizer.featurize(data_set)
        features.to_csv(feature_filename)
        summary = features.describe(include='all')
        summary.to_csv(f"{'.'.join(feature_filename.split('.')[:-1])}.csv")
    else:
        print("Loading features...")
        features = pd.read_csv(feature_filename)
    return data_set.merge(features, left_index=True, right_index=True)


if __name__ == "__main__":
    main()
