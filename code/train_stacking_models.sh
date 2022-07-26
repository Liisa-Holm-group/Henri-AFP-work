#!/bin/bash

DATA_PATH=$1
OUTPUT_PATH=$1

for MODEL in "LTR_xgb" "mean" "ranking_mean" "svm" "sgdc" "ann"
do
    for ONTOLOGY in "CC" "MF" "BP"
    do
        echo "training $ONTOLOGY models"
        python train_stacking_models.py "${MODEL}_${ONTOLOGY}_stacking" "${MODEL}_stacking" ${DATA_PATH} "${DATA_PATH}/${ONTOLOGY}_targets.npz" $OUTPUT_PATH  ${ONTOLOGY} 1
    done
done 
