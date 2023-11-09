import os

import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from featurizer import Featurizer
from classifier import Classifier


RERUN_FEATURES = False


def main():
    # data_dir = (os.path.join(os.getcwd(), "Assignments", "Assignment 4", "data"))
    
    featurizer = Featurizer()

    full_train_set = get_features(featurizer, RERUN_FEATURES, feature_filename="data/train_features.csv", filename="data/pan2324_train_data.csv")
    full_dev_set = get_features(featurizer, RERUN_FEATURES, feature_filename="data/dev_features.csv", filename="data/pan2324_dev_data.csv")
    # full_test_set = get_features(featurizer, RERUN_FEATURES, feature_filename="data/test_features.csv", filename="data/pan2324_test_data.csv")

    print("Running classifier...")
    classifier = Classifier()

    colnames = [col for col in full_train_set.columns if col.startswith("f_")]
    
    train_x = full_train_set[colnames].to_numpy()
    train_y = full_train_set["author"].to_numpy()
    classifier.fit(train_x, train_y)

    print("Evaluating classifier...")
    dev_x = full_dev_set[[col for col in full_train_set.columns if col.startswith("f_")]].to_numpy()
    dev_y = full_dev_set["author"].to_numpy()
    classifier.evaluate(dev_x, dev_y)

    # Ablation
    print("Starting ablation study")
    ablation_colname_sets = []
    for i in range(len(colnames)):
        ablation_colname_sets.append([colname for j, colname in enumerate(colnames) if i!=j])
    
    ablation_classifiers = {colname: Classifier() for colname in colnames}
    
    print("Training classifiers...")
    ablation_train_sets = {colnames[i]: full_train_set[ablation_colnames].to_numpy() for i, ablation_colnames in enumerate(ablation_colname_sets)}
    [ablation_classifier.fit(train_x=train_set, train_y=train_y) for (missing_feat_name, train_set), ablation_classifier in zip(ablation_train_sets.items(), ablation_classifiers.values())]
    ablation_train_results = {missing_feat_name: ablation_classifier.get_f1_score(eval_x=train_set, eval_y=train_y) for (missing_feat_name, train_set), ablation_classifier in zip(ablation_train_sets.items(), ablation_classifiers.values())}
    plot_ablation_results(ablation_train_results, name="Train")

    
    print("Evaluating classifiers...")
    ablation_dev_sets = {colnames[i]: full_dev_set[ablation_colnames].to_numpy() for i, ablation_colnames in enumerate(ablation_colname_sets)}
    ablation_results = {missing_feat_name: ablation_classifier.get_f1_score(eval_x=dev_set, eval_y=dev_y) for (missing_feat_name, dev_set), ablation_classifier in zip(ablation_dev_sets.items(), ablation_classifiers.values())}
    plot_ablation_results(ablation_results, name="Train")

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


def plot_ablation_results(ablation_results: dict, name: str = ""):
    ablation_results = dict(sorted(ablation_results.items()))
    fig = plt.figure(figsize = (10, 5))
    x_ticks = [tick[2:] for tick in ablation_results.keys()]
    plt.bar(x_ticks, ablation_results.values(), width = 0.4)
    plt.xlabel("Missing feature")
    plt.ylabel("F1 score")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.title(f"{name} - Ablation study for authorship attribution")
    plt.show()
    plt.savefig(f"ablation_plot_{name}.png")
    

if __name__ == "__main__":
    main()
