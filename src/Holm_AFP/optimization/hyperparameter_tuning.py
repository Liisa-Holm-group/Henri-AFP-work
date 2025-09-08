import optuna
from Holm_AFP.optimization.CV_split import hyperparameter_run
from Holm_AFP.optimization.optimization_settings import ModelName, OptimizationSettings


def objective(trial: optuna.Trial) -> float:


    number_of_hits = trial.suggest_int("number_of_hits", 1, 10)
    number_of_neighbours = trial.suggest_int("number_of_neighbours", 1, 20)
    model_name = trial.suggest_categorical("model_name", ModelName.get_trainable_models())

    additional_argument_dict = {}

    n_evaluation_jobs = 4

    n_prediction_jobs = 32
    n_training_jobs = 32

    if model_name.value == "cnn":
        additional_argument_dict["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        n_training_jobs = 20
    if model_name == ModelName.XGB:
        additional_argument_dict.update({
        # learning dynamics
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "max_delta_step": trial.suggest_float("max_delta_step", 0.0, 10.0),

        # tree complexity
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 2.0, 20.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 10.0),

        # sampling
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),

        # regularization
        "alpha": trial.suggest_float("alpha", 1e-8, 100.0, log=True),
        "lambda": trial.suggest_float("lambda", 1e-3, 100.0, log=True),

        # fixed best-practice defaults
        "tree_method": "hist",
        "eval_metric": "aucpr",
        })


    trial_optimization_settings = OptimizationSettings(
        model_name=model_name,
        number_of_hits=number_of_hits,
        number_of_neighbours=number_of_neighbours,
        n_training_jobs=n_training_jobs,
        n_prediction_jobs=n_prediction_jobs,
        n_evaluation_jobs=n_evaluation_jobs,
        **additional_argument_dict
    )

    return hyperparameter_run(optimization_settings=trial_optimization_settings)


if __name__ == "__main__":
    study = optuna.create_study(study_name="xgboost_test_3_9",
                                storage="sqlite:///db.sqlite3"
                                )
    study.optimize(objective, n_trials=3)
    print(f"Best value: {study.best_value} (params: {study.best_params})")
    pass


