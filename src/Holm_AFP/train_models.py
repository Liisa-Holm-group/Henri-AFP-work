"""Train models on full data."""

import argparse
import numpy as np
import models

np.random.seed(1)


def main():
    """CLI for training models."""
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name')
    parser.add_argument('model')
    parser.add_argument('go_class_names', help='MF, CC, or BP')
    parser.add_argument('tr_feature_path')
    parser.add_argument('tr_target_path')
    parser.add_argument('feature_names')
    parser.add_argument('output_path')
    parser.add_argument('n_jobs', type=int)
    parser.add_argument('-d', '--debug', action='store_true',
                        help='debug mode: run the experiment for only 20 classes')

    args = parser.parse_args()

    model_trainer = models.ModelTrainer(args.experiment_name, getattr(models, args.model),
                                        args.go_class_names, args.tr_feature_path,
                                        args.tr_target_path, args.feature_names, args.output_path)

    model_trainer.run(n_jobs=args.n_jobs)

if __name__ == '__main__':
    main()
