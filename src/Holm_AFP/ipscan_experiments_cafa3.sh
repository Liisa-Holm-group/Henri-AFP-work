#!/bin/bash

# cafa3 cross validation runs:
for ONTOLOGY in "CC" "MF" "BP"
    do
    for feature_set in 1 2 3
    do
        for model in "xgboost" "lasso" "fm" "svm"
do


    NAMES="$1/combined_cafa3_cc/old/${ONTOLOGY}_cafa3_feature_names.joblib"
    OUTPUT="$2/cv_results"

    if [ "$ONTOLOGY" = "CC" ]; then
        FEATURES="$1/combined_cafa3_cc/${ONTOLOGY}_cafa3_features.npz"
        TARGETS="$1/combined_cafa3_cc/${ONTOLOGY}_targets.npz"
    else
        FEATURES="$1/cafa3_data/datasets/${ONTOLOGY}_cafa3_features.npz"
        TARGETS="$1/cafa3_data/datasets/${ONTOLOGY}_targets.npz"
    fi
    python ipscan_experiment.py "CAFA3_CC_${model}" $feature_set "${model}_test" $FEATURES $TARGETS $NAMES $OUTPUT 40
done
done
done




