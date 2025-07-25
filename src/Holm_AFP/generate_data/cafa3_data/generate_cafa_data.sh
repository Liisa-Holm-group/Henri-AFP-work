#!/bin/bash

DATA_PATH=/data/henri/CAFA_FeatureData
OUTPUT_PATH=/data/henri/cafa_datasets

python load_data.py $DATA_PATH $OUTPUT_PATH MF ips_plus_ta -p &
python load_data.py $DATA_PATH $OUTPUT_PATH BP ips_plus_ta -p &
python load_data.py $DATA_PATH $OUTPUT_PATH CC ips_plus_ta -p &

python load_data.py $DATA_PATH $OUTPUT_PATH MF sans_plus_others -p &
python load_data.py $DATA_PATH $OUTPUT_PATH BP sans_plus_others -p &
python load_data.py $DATA_PATH $OUTPUT_PATH CC sans_plus_others -p &

python load_data.py $DATA_PATH $OUTPUT_PATH MF string -p &
python load_data.py $DATA_PATH $OUTPUT_PATH BP string -p &
python load_data.py $DATA_PATH $OUTPUT_PATH CC string -p &
