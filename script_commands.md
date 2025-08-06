## Script commands



Full use of the AFP pipeline requires for us to:

1. Train a model
2. Predict with the model
3. Evaluate the predictions

I will show how to get these parts to run in this tutorial on the Mahti server.

Note: The commands listed below are expected to be run from the [source](./src/Holm_AFP) directory.
----

#### 1. train a model

Here is an input to trian a model using Lasso on a dataset with only 300 GO classes
```{bash}
train_models.py
lasso_predict
/scratch/project_2008455/Henri-AFP-work/results/lasso_max_old_data_train_string_search_full_models.joblib
/scratch/project_2008455/Max_temp/diamond_search/data/training_data/old_data_300_go_terms/truth_go_list.joblib
/scratch/project_2008455/Max_temp/diamond_search/data/training_data/old_data_300_go_terms/X_sparse.npz
/scratch/project_2008455/Max_temp/diamond_search/data/training_data/old_data_300_go_terms/gene_list.joblib
../../results
/scratch/project_2008455/Max_temp/diamond_search/data/training_data/old_data_300_go_terms/feature_list.joblib
```

The model results are saved in the root directory in the directory ```/results```

---

#### 2.  Predict with the model

Next we create model 

Note: We try to predict on the same data we used for training in this example

```{bash}
predict.py
lasso_predict
/scratch/project_2008455/Henri-AFP-work/results/lasso_max_old_data_train_string_search_full_models.joblib
/scratch/project_2008455/Max_temp/diamond_search/data/training_data/old_data_300_go_terms/truth_go_list.joblib
/scratch/project_2008455/Max_temp/diamond_search/data/training_data/old_data_300_go_terms/X_sparse.npz
/scratch/project_2008455/Max_temp/diamond_search/data/training_data/old_data_300_go_terms/gene_list.joblib
../../results
/scratch/project_2008455/Max_temp/diamond_search/data/training_data/old_data_300_go_terms/feature_list.joblib
```

---

#### 3. Predict with model results

This part does not fully work. Firstly, you need to copy Henri's evaluation scripts from the data directory listen in the README.md. 
The script to run will be in the data directory in this path ```/scratch/project_2008455/Henri-AFP-work/data2/data/cafa3_data/cafa3/CAFA_MatlabCode/CAFA2-master/MyAnalysisScripts/CAFA_analysis_pipe.m``

The pipeline, however, contains hard-coded variables that need to be changes to match our curent obo file. Even with this, the evaluation does not work for me.
There is something wrong in the obo reading section. If it was to work, this is how to run the script
```
matlab -nodisplay -nodesktop -r \
  "CAFA_analysis_pipe('/scratch/project_2008455/Henri-AFP-work/results/lasso_predict_new_custom_predictions','/scratch/project_2008455/Henri-AFP-work/src/Holm_AFP/generate_data/evaluation_table/stacked_results.tsv','/scratch/project_2008455/Henri-AFP-work/src/Holm_AFP/generate_data/evaluation_table/eval_sequences.txt','BP','/scratch/project_2008455/Henri-AFP-work/final_results/evaluation/lasso_predict_new_custom_prediction_cafa_scores.txt'); quit force" \
  >> tmp.txt
```
