#!/bin/bash

DATA_PATH=$1
OUTPUT_PATH=$1


for MODEL in "LTR_xgb" "mean" "ranking_mean" "svm" "sgdc" "ann"
do
    for ONTOLOGY in "CC" "MF" "BP"
    do

        if [ "$ONTOLOGY" = "CC" ]; then
            additional_features="$1/combined_cafa3_cc/${ONTOLOGY}_stacking_features"
            taxonomy_features="$1/combined_cafa3_cc/${ONTOLOGY}_stacking_taxonomy.npz"
        else
            additional_features="${OUTPUT_PATH}/cafa3_data/datasets/${ONTOLOGY}_stacking_features.npy"
            taxonomy_features="${OUTPUT_PATH}/cafa3_data/datasets/${ONTOLOGY}_stacking_taxonomy.npz"
        fi
        echo "training $ONTOLOGY models"
        python train_stacking_models.py "${MODEL}_${ONTOLOGY}_stacking" "${MODEL}_stacking" ${DATA_PATH} "${DATA_PATH}/${ONTOLOGY}_targets.npz" $OUTPUT_PATH  ${ONTOLOGY} 1 $additional_features $taxonomy_features
    done
done 
