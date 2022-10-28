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
    parser.add_argument('predictions_dir')
    parser.add_argument('tr_target_path')
    parser.add_argument('output_path')
    parser.add_argument('ontology')
    parser.add_argument('n_jobs', type=int)

    parser.add_argument('additional_features', type=str)
    parser.add_argument('taxonomy_features', type=str)

    parser.add_argument('-d', '--debug', action='store_true',
                        help='debug mode: run the experiment for only 20 classes')
    parser.add_argument('--level3', action='store_true', help='train level3 models')

    args = parser.parse_args()

    model_trainer = models.StackingModelTrainer(args.experiment_name, getattr(models, args.model),
                                        args.predictions_dir, args.tr_target_path, args.output_path, args.ontology, args.additional_features, args.taxonomy_features, args. args.level3)

    model_trainer.run(n_jobs=args.n_jobs, debug=args.debug)

if __name__ == '__main__':
    main()
