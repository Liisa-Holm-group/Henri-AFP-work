#!/bin/bash

for ONTOLOGY in "CC" "MF" "BP"
    do
    for MODEL in "xgboost" "lasso" "svm"
    do
    for feature_set in 1 2 3
        do
        FEATURES="$1/datasets/BP_ipscan_features.npz" 
        NAMES="$1/datasets/BP_ipscan_feature_names.joblib" 
        TARGETS="$1/datasets/BP_targets.npz" 
        OUTPUT="$2" 
        python ipscan_experiment.py "${ONTOLOGY}_${MODEL}" $feature_set "${MODEL}_test" $FEATURES $TARGETS $NAMES $OUTPUT 40
    done
done
done



