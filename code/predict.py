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
    parser.add_argument('go_class_names', help='MF, CC, or BP')
    parser.add_argument('te_feature_path')
    parser.add_argument('te_sequences')
    parser.add_argument('output_path')
    parser.add_argument('feature_names')
    parser.add_argument('--h5', action='store_true', help='save results in h5 format')
    args = parser.parse_args()

    predictor = models.Predictor(args.experiment_name, args.model_path,
                                 args.go_class_names, args.te_feature_path,
                                 args.te_sequences, args.output_path, args.feature_names, args.h5)
    predictor.run()

if __name__ == '__main__':
    main()
