#!/bin/bash

OUTPUT_PATH="$2"

for DATA_PATH in "$1/cafa3_data/only_ipscan_datasets/" "$1/cafa3_data/datasets/"
do
for MODEL in "lasso" "xgb" "svm" "fm"
do
    for ONTOLOGY in "CC" "MF" "BP"
    do
        echo "training $ONTOLOGY models"


    if [[ "$DATA_PATH" == *"ipscan"* ]]; then
        FEATURES="$1/combined_cafa3_cc/${ONTOLOGY}_cafa3_features.npz"
        TARGETS="$1/combined_cafa3_cc/${ONTOLOGY}_targets.npz"

        python train_models.py "${MODEL}_${ONTOLOGY}" "${MODEL}_train" "${DATA_PATH}/${ONTOLOGY}_target_names.joblib" "${DATA_PATH}/${ONTOLOGY}_cafa3_only_ipscan_features.npz" "${DATA_PATH}/${ONTOLOGY}_targets.npz" "${DATA_PATH}/${ONTOLOGY}_cafa3_only_ipscan_feature_names.joblib" $OUTPUT_PATH 50 1
    else
        FEATURES="$1/cafa3_data/datasets/${ONTOLOGY}_cafa3_features.npz"
        TARGETS="$1/cafa3_data/datasets/${ONTOLOGY}_targets.npz"
    fi

        python train_models.py "${MODEL}_${ONTOLOGY}" "${MODEL}_train" "${DATA_PATH}/${ONTOLOGY}_target_names.joblib" "${DATA_PATH}/${ONTOLOGY}_cafa3_features.npz" "${DATA_PATH}/${ONTOLOGY}_targets.npz" "${DATA_PATH}/${ONTOLOGY}_cafa3_feature_names.joblib" $OUTPUT_PATH 50 0
        

    done
done 
done
