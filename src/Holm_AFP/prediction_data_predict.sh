#!/bin/bash


OUTPUT_PATH=$2

for model in "lasso" "svm" "fm" "xgb"
do
for feature in "ipscan" "taxonomy" "location"
do
    for ONTOLOGY in "CC" "MF" "BP"
    do

    if [ "$feature" == "ipscan" ]; then

        DATA_PATH="$1/cafa3_data/CAFA3_Features_EvalSet/only_ipscan_datasets/"
        python predict.py "${model}_${ONTOLOGY}" "${OUTPUT_PATH}/${model}_${ONTOLOGY}_${feature}_full_models.joblib" "${DATA_PATH}/${ONTOLOGY}_target_names.joblib" "${DATA_PATH}/CAFA_${ONTOLOGY}_cafa3_eval_data_only_ipscan_features.npz" "${DATA_PATH}/CAFA_${ONTOLOGY}_sequences.joblib" $OUTPUT_PATH "${DATA_PATH}/CAFA_${ONTOLOGY}_cafa3_eval_data_only_ipscan_feature_names.joblib"
    else
        DATA_PATH="$1/cafa3_data/CAFA3_Features_EvalSet/new_datasets" #NOTE from biotek-groups2
        python predict.py "${model}_${ONTOLOGY}" "${OUTPUT_PATH}/${model}_${ONTOLOGY}_${feature}_full_models.joblib" "${DATA_PATH}/${ONTOLOGY}_target_names.joblib" "${DATA_PATH}/CAFA_${ONTOLOGY}_ipscan_features.npz" "${DATA_PATH}/CAFA_${ONTOLOGY}_sequences.joblib" $OUTPUT_PATH "${DATA_PATH}/CAFA_${ONTOLOGY}_ipscan_feature_names.joblib"
    fi

done
done
