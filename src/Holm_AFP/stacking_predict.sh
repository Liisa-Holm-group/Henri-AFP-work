#!/bin/bash

DATA_PATH=$1
OUTPUT_PATH=$1

TR_PATH=$1
for model in "sgdc" "svm" "ann" "mean" "xgb" "LTR_xgb" 
do
    for ONTOLOGY in "CC" "MF" "BP"
    do

        if [ "$ONTOLOGY" = "CC" ]; then
            additional_features="$1/combined_cafa3_cc/${ONTOLOGY}_stacking_features_eval.npy"
            taxonomy_features="$1/combined_cafa3_cc/${ONTOLOGY}_stacking_taxonomy.npz"
        else
            additional_features="${OUTPUT_PATH}/cafa3_data/new_results/${ONTOLOGY}_stacking_features_eval.npy"
            taxonomy_features="${OUTPUT_PATH}/cafa3_data/new_results/${ONTOLOGY}_stacking_taxonomy.npz"
        fi
        python stacking_predict.py "${model}_${ONTOLOGY}" "${DATA_PATH}/${model}_${ONTOLOGY}_stacking_${ONTOLOGY}_stacking_models.joblib" "${DATA_PATH}/${ONTOLOGY}_target_names.joblib" "${DATA_PATH}" "${DATA_PATH}/CAFA_${ONTOLOGY}_sequences.joblib" $OUTPUT_PATH $ONTOLOGY $additional_features $taxonomy_features 60 --tr_predictions $TR_PATH
    done
done
