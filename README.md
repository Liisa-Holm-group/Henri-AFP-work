This code represents the code that Henri Tiittanen did in our group on protein function prediction (Automated Function Prediction, AFP). Link to preprint that describes the work and results in detail is here:

https://www.biorxiv.org/content/10.1101/2022.08.10.503467v1

This work used features from InterProScan (IPS), predictions on cellular localization and species taxonomy as input features. Main emphasis was on InterProScan features. Our aim was to look for thhe best ways to process IPS features. We compared binary features vs. features with IPS signal strength. We tested IPS features with rough localization information, IPS features with added occurrence counts, IPS features with IPS clusters (=grouping generated by IPS) and IPS features with species taxonomy.

Work was the first ones to test eXtreme Gradient Boosting (XGB) algorithm for AFP. Work is still onlty article that demonstrates Factorization Machines (FMs) for function prediction. Other classifiers were SVM and Elastic Net (sparse linear regression). Work is still only article that demonstrates the classifier stacking where the stacking optimization is performed separately for each GO class. This allows the different classifier optimization for each GO class. We argue that it generates significant improvement in results.

Work represents code for running massive number of classifiers (N*1000 classes, 4 classifiers, different input types) with 5-fold Cross Validation. 5-fold CV is required for classifier stacking. It represents code for processing IPS features. It represents code for running classifiers like XGB, FM and SVM. All the runs were done on kronos server.

Why this work?

This feature dataset will provide a good reference point for other methods (PPI, embedding). In addition, this feature set might useful, if we decide to combine new feature sets with this data. We could see look for functional classes that can predicted better with, say a combination of sequence embeddings and IPS. These combinations could lead us to new theories on what causes protein 

ToDo and problems

I list here things that should be done differently with this work. 

1. Project created a separate stratified 5-fold CV for each GO class. This caused problems, as we could not combine the CV-runs between GO classes. We generated later a specific code to create stratified CV-splits across multiple classes with strong class imbalance (Tiittanen, optisplit) 
2. Project did not have a separate training and evaluation data. All the input data was used in the 5-fold CV. There should be separate evaluation and test dataset, outside the CV run. These can be used to check the performance of stacking classifiers.
3. Parameter optimization should be done in a better way. There was no separate evaluation data for hyperparameter optimization. We did not use Bayesian hyperparameter optimization at the time.
3.2. Hao has a code that performs Bayesian hyperparameter optimization for XGB, SVM etc.
4. Data that project used, included curated predictions. These generate circular logic (data leakage) risk.
5. Project used only binary classifiers. One should consider multi-label classifiers (pyBoost for example).


Below is the formatted original README.md instructions by Henri
---

# Download data

### Data can be downloaded from the project web page [ekhidna2.biocenter.helsinki.fi/AFP](ekhidna2.biocenter.helsinki.fi/AFP)

### Get CAFA evaluation codes from [CAFA 2 GitHub yuxjiang ](https\://github.com/yuxjiang/CAFA2)

### copy that repository to `DATA\_DIRECTORY\_PATH/cafa3\_data/cafa3/CAFA\_MatlabCode`

### move `DATA\_DIRECTORY\_PATH/cafa3\_data/cafa3/MyAnalysisScripts directory to`

### `DATA\_DIRECTORY\_PATH/cafa3\_data/cafa3/CAFA\_MatlabCode`

### Follow the steps below to run the experiments:

## 1. Create virtual environment and install dependencies

```bash
python3 -m venv .env && source .env/bin/activate && pip install -r requirements.txt
```

## 2. Generate in-house data cross validated predictions

```bash
bash ipscan_experiments.sh DATA_DIRECTORY_PATH OUTPUT_PATH
# Example:
bash ipscan_experiments.sh ../data2/data ../results
```

## 3. Generate in-house data stacking predictions

```bash
bash stacking_experiment.sh DATA_DIRECTORY_PATH OUTPUT_PATH
```

## 4. Evaluate in-house data CV predictions

```bash
python process_results.py OUTPUT_PATH
```

## 5. Train CAFA3 first level models on full CAFA training data

```bash
bash train_models.sh DATA_DIRECTORY_PATH OUTPUT_PATH
# Example:
bash train_models.sh ../data2/data ../results
```

## 6. Generate CAFA3 first level predictions for CAFA3 test data

```bash
bash predict.sh DATA_DIRECTORY_PATH
```

## 7. Generate CAFA3 cross validated predictions for training the stacking models

```bash
bash ipscan_experiments_cafa3.sh DATA_DIRECTORY_PATH OUTPUT_PATH
```

## 8. Train CAFA3 stacking models on 1st level predictions

```bash
bash train_stacking_models.sh OUTPUT_PATH
```

## 9. Generate CAFA3 stacking predictions for CAFA3 test data

```bash
bash stacking_predict.sh OUTPUT_PATH
```

## 10. Evaluate CAFA3 predictions

```bash
cd cafa3_data/cafa3/CAFA_MatlabCode/CAFA2-master/MyAnalysisScripts
bash evaluate_predictions.sh DATA_DIRECTORY_PATH OUTPUT_PATH
```

## 11. Evaluate CAFA3 stacking predictions

```bash
cd cafa3_data/cafa3/CAFA_MatlabCode/CAFA2-master/MyAnalysisScripts
bash evaluate_predictions.sh DATA_DIRECTORY_PATH OUTPUT_PATH
```

