#!/bin/bash

DATA_PATH=/data/henri/
OUTPUT_PATH=/data/henri/datasets_NEW

python cafa_load_data.py $DATA_PATH $OUTPUT_PATH CC ipscan 
python load_data.py $DATA_PATH $OUTPUT_PATH MF ipscan 
python load_data.py $DATA_PATH $OUTPUT_PATH BP ipscan 
