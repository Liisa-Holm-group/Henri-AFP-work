#!/bin/bash

for model in "fm" "svm" "lasso" "xgb"
do
for feature in "taxonomy" "location"
do
    for ONTOLOGY in "CC" "MF" "BP"
    do

        if [ "$feature" = "ipscan" ]; then
            DATA_PATH="$1/cafa3_data/CAFA3_Features_EvalSet/only_ipscan_datasets/"
            OUTPUT_PATH="$1/cafa3_data/only_ipscan_results"

            python predict.py "${model}_${ONTOLOGY}" "${OUTPUT_PATH}/${model}_${ONTOLOGY}_${feature}_full_models.joblib" "${DATA_PATH}/${ONTOLOGY}_target_names.joblib" "${DATA_PATH}/CAFA_${ONTOLOGY}_cafa3_eval_data_only_ipscan_features.npz" "${DATA_PATH}/CAFA_${ONTOLOGY}_sequences.joblib" $OUTPUT_PATH "${DATA_PATH}/CAFA_${ONTOLOGY}_cafa3_eval_data_only_ipscan_feature_names.joblib" &
        else
            DATA_PATH="$1/cafa3_data/CAFA3_Features_EvalSet/new_datasets/"
            OUTPUT_PATH="$1/cafa3_data/new_results"

            python predict.py "${model}_${ONTOLOGY}" "${OUTPUT_PATH}/${model}_${ONTOLOGY}_${feature}_full_models.joblib" "${DATA_PATH}/${ONTOLOGY}_target_names.joblib" "${DATA_PATH}/CAFA_${ONTOLOGY}_cafa3_eval_data_features.npz" "${DATA_PATH}/CAFA_${ONTOLOGY}_sequences.joblib" $OUTPUT_PATH "${DATA_PATH}/CAFA_${ONTOLOGY}_cafa3_eval_data_feature_names.joblib" --h5 
        fi

    done
done
done
