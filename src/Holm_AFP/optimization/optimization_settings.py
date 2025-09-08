from enum import StrEnum
from typing import Optional, List
from pydantic import BaseModel, field_validator


class ModelName(StrEnum):
    LASSO = "lasso_train"
    XGB = "xgb_train"
    ELASTICNET = "elasticnet_train"
    DUMMY_TRAIN = "dummy_train"

    @staticmethod
    def get_trainable_models() -> List["ModelName"]:
        return [ModelName.XGB]
        # return [ModelName.LASSO, ModelName.XGB, ModelName.ELASTICNET]

class OptimizationSettings(BaseModel):
    model_name: ModelName
    number_of_hits: int
    number_of_neighbours: int
    learning_rate: Optional[float] = None
    optimizer: Optional[str] = None
    n_training_jobs: int
    n_prediction_jobs: int
    n_evaluation_jobs: int

    @field_validator("number_of_hits")
    @classmethod
    def _hits(cls, v):
        if v < 1: raise ValueError("Number of hits must be >= 1")
        return v

    @field_validator("number_of_neighbours")
    @classmethod
    def _neigh(cls, v):
        if v < 1: raise ValueError("Number of neighbours must be >= 1")
        return v

    @field_validator("learning_rate")
    @classmethod
    def _lr(cls, v):
        if v is not None and not (0 < v <= 1):
            raise ValueError("learning_rate must be in (0,1]")
        return v

    @field_validator("optimizer")
    @classmethod
    def _opt(cls, v):
        if v is not None and v not in {"adam", "sgd", "rmsprop"}:
            raise ValueError(f"Invalid optimizer: {v}")
        return v

    @field_validator("n_training_jobs", "n_prediction_jobs", "n_evaluation_jobs")
    @classmethod
    def _n_workers(cls, v):
        if v < 1 or not isinstance(v, int): raise ValueError("Number of jobs be >= 1 and an integer")
        return v

    def number_of_meta_settings(self):
        return self.number_of_hits*5