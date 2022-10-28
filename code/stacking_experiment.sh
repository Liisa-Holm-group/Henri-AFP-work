#!/bin/bash

output_path=$2
target_path=$1/datasets
level2_data_path="/home/biotek-groups2/holm/henri/ipscan_results/"
level3_data_path=' '
cluster_index_path=$1
n_jobs=60
level3='False'
ranking='False'
taxonomy='False'
pool='False'

if [ ! -d "../results/datasets" ]; then
   mkdir "../results/datasets"
fi
cp ../data2/data/datasets/??_stacking_* ../results/datasets

for additional in 'True' 'False'
do
for model in 'xgb_stacking' 'ann_stacking' 'LTR_xgb_stacking' 'sgdc_stacking'
do
    for ontology in 'CC' 'MF' 'BP'
    do
        python stacking_experiment.py $model $output_path $target_path $level3_data_path $level2_data_path $cluster_index_path $ontology $n_jobs $level3 $ranking $additional $taxonomy $pool -d
        # echo stacking_experiment.py $model $output_path $target_path $level3_data_path $level2_data_path $cluster_index_path $ontology $n_jobs $level3 $ranking $additional $taxonomy $pool
    done
done
done
