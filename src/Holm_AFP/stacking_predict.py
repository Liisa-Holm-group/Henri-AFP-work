"""Predict on new data using pretrained models."""

import argparse

import numpy as np

import models

np.random.seed(1)

def main():
    """CLI for predicting with pretrained models."""
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name')
    parser.add_argument('model_path', help='location of the pretrained models')
    parser.add_argument('go_class_names')
    parser.add_argument('te_feature_dir')
    parser.add_argument('te_sequences')
    parser.add_argument('output_path')
    parser.add_argument('ontology')
    parser.add_argument('additional_features')
    parser.add_argument('stacking_features')
    parser.add_argument('n_jobs')
    parser.add_argument('--tr_predictions', default=None)
    args = parser.parse_args()

    predictor = models.StackingPredictor(args.experiment_name, args.model_path,
            args.go_class_names, args.te_feature_dir, args.te_sequences,
            args.output_path, args.ontology, args.additional_features, args.taxonomy_features, args.n_jobs,
            args.tr_predictions,)
    predictor.run()

if __name__ == '__main__':
    main()
