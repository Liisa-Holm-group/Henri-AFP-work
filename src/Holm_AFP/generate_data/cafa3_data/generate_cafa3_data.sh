#!/bin/bash


# Generate cafa3 training data:
DATA_PATH=/data/henri/cafa3_data/
OUTPUT_PATH=/data/henri/cafa3_data_NEW

python cafa3_load_data2.py $DATA_PATH $OUTPUT_PATH CC cafa3 &
python cafa3_load_data2.py $DATA_PATH $OUTPUT_PATH MF cafa3 &
python cafa3_load_data2.py $DATA_PATH $OUTPUT_PATH BP cafa3 &

wait

#Generate evaluation sets:
DATA_PATH=/data/henri/cafa3_data/CAFA3_Features_EvalSet/
OUTPUT_PATH=/data/henri/cafa3_data_NEW/

python cafa3_load_data2.py $DATA_PATH $OUTPUT_PATH CC cafa3_eval_data -p &
python cafa3_load_data2.py $DATA_PATH $OUTPUT_PATH MF cafa3_eval_data -p &
python cafa3_load_data2.py $DATA_PATH $OUTPUT_PATH BP cafa3_eval_data -p &
