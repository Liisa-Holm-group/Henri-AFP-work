from typing import Dict, List
import shutil
import joblib
import numpy as np
import pandas as pd
from cafaeval.evaluation import cafa_eval, write_results
from diamond_search.data_forming.add_inherited_terms_to_protein_sequences import add_propagated_results_to_joblib_matrix
from diamond_search.data_forming.diamond_complete import get_string_neighbourhood_dataset, \
    save_result_df_to_sparse_with_joblib, merge_diamond_results_with_ground_truth
from diamond_search.utils.split_single_joblib_for_training import filter_data_and_save_to_separate_files, \
    save_results
from scipy.sparse import load_npz, vstack
from pathlib import Path
from Holm_AFP.optimization.config import  (CV_JOBLIB_PATH, DIAMOND_EXECUTABLE_PATH, DIAMOND_FILE_ROOT_DIR,
                                           STRING_LINKS_PATH, STRING_ENRICHMENT_PATH, OBO_FILE_PATH, GROUND_TRUTH_PATH)
from Holm_AFP import models
from logging import getLogger
import ia

from Holm_AFP.optimization.optimization_settings import OptimizationSettings, ModelName

logger = getLogger(__name__)

def add_fold_index_to_STRING_joblib_by_taxon_to_fold_dict(taxon_name_fold_map: Dict[str, int], joblib_path, output_joblib_path: str):
    """ Adds folds to the joblib results. Saves the new joblib to new joblib path"""

    logger.info(f"Adding folds to {output_joblib_path}")
    string_result_joblib = joblib.load(joblib_path)

    new_joblib_fold_list = [taxon_name_fold_map.get(string_name, None) for string_name in string_result_joblib["taxon"]]

    string_result_joblib["gene_folds"] = new_joblib_fold_list

    joblib.dump(string_result_joblib, output_joblib_path)



def make_list_unique(seq: List):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]



def split_joblib_to_fold_dirs(joblib_path: str, output_dir_prefix: str):
    """Split the joblib by gene folds into separate directories and files"""

    logger.info(f"Splitting {joblib_path}")
    string_result_joblib = joblib.load(joblib_path)

    folds = string_result_joblib["gene_folds"]
    fold_names = []
    for fold_idx in make_list_unique(folds):
        if fold_idx is None:
            continue
        logger.info(f"Fold {fold_idx}")
        fold_indexes = [idx for idx, val in enumerate(folds) if val == fold_idx]
        file_name = f"{output_dir_prefix}_{fold_idx}"
        filter_data_and_save_to_separate_files(joblib_path, output_dir=file_name, gene_indexes=fold_indexes)
        fold_names.append(file_name)
    return fold_names


def construct_data(optimization_settings: OptimizationSettings, obo_path: str,
                   ground_truth_joblib_path: str, cv_joblib_path: str):
    """ Construct CV data for training. NOTE: The ground truth .fasta files must be constructed before calling this function!"""

    # Get original data
    diamond_result_path = f"{DIAMOND_FILE_ROOT_DIR}/temp_files/string_search_results.tsv"
    diamond_datafile_path = f"{DIAMOND_FILE_ROOT_DIR}/string12.dmnd"
    prefix = "CV_old"
    diamond_result_joblib_path = f"{DIAMOND_FILE_ROOT_DIR}/temp_files/{prefix}_sequence_array.joblib"
    propagated_diamond_joblib_path = f"{DIAMOND_FILE_ROOT_DIR}/temp_files/{prefix}_propagated_sequence_array.joblib"
    merged_joblib_file_path = f"{DIAMOND_FILE_ROOT_DIR}/data/training_data/{prefix}_full.joblib"
    output_dir = f"{DIAMOND_FILE_ROOT_DIR}/data/training_data/split_CV/{prefix}_data_CV_split"
    data_with_CV_indexes_path = f"{DIAMOND_FILE_ROOT_DIR}/data/training_data/{prefix}_data_target_jobs"

    logger.info(f"Constructing {diamond_result_path}")
    sequence_data = get_string_neighbourhood_dataset(
                                                     diamond_path=DIAMOND_EXECUTABLE_PATH,
                                                     query_tsv_file=STRING_ENRICHMENT_PATH,
                                                     string_links_path=STRING_LINKS_PATH,
                                                     diamond_result_path=diamond_result_path,
                                                     diamond_datafile_path=diamond_datafile_path,
                                                     number_of_hits=optimization_settings.number_of_hits,
                                                     string_enrichment_terms_path=STRING_ENRICHMENT_PATH,
                                                     top_neighbour_count=optimization_settings.number_of_neighbours)


    logger.info("Save sparse to joblib")
    save_result_df_to_sparse_with_joblib(sequence_data, diamond_result_joblib_path, number_of_hits=optimization_settings.number_of_hits)
    logger.info("Propagate child results to parent results")
    add_propagated_results_to_joblib_matrix(diamond_result_joblib_path, propagated_diamond_joblib_path, go_dag_path=obo_path)



    logger.info("Merge with ground truth")
    merge_diamond_results_with_ground_truth(diamond_results_joblib_path=propagated_diamond_joblib_path,
                                            ground_truth_joblib_path=ground_truth_joblib_path, data_prefix="old",
                                            output_joblib_path=merged_joblib_file_path)


    logger.info("Split data by CV fold into separate directories")
    # Get CV split arguments
    cv_split_data = joblib.load(cv_joblib_path)
    clusters = cv_split_data["data_splits"]
    cluster_dict = {idx: cluster_idx for cluster_idx, cluster in enumerate(clusters) for idx in cluster}
    taxon_to_cluster_name = {taxon_name: cluster_dict[taxon_index] for taxon_name, taxon_index in cv_split_data["cluster_lookup_dict"].items()}

    add_fold_index_to_STRING_joblib_by_taxon_to_fold_dict(taxon_to_cluster_name, merged_joblib_file_path, data_with_CV_indexes_path)
    # Save each fold then to a separate joblib file or just add indexes into the joblib file?
    fold_dirs = split_joblib_to_fold_dirs(data_with_CV_indexes_path, output_dir)

    return fold_dirs


def construct_cv_datafile(list_of_dirs: List[str], output_dir: str):
    """ Join the files with the same name together. Assumes that the directory data only differs in the protein list and their
    impact on X and y
    Custom logic is needed for input features and sparse array joining
    """

    logger.debug("Constructing CV datafile")
    logger.debug(f"List of directories: {list_of_dirs}")
    logger.debug(f"Output dir: {output_dir}")
    truth_go_list = joblib.load(f"{list_of_dirs[0]}/truth_go_list.joblib")
    feature_list = joblib.load(f"{list_of_dirs[0]}/feature_list.joblib")
    X_sparse_list = []
    y_sparse_list = []
    gene_list = []
    taxon = []
    gene_dates = []
    for cv_dir in list_of_dirs:
        logger.debug(f"CV dir: {cv_dir}")
        X_sparse_list.append(load_npz(f"{cv_dir}/X_sparse.npz"))
        y_sparse_list.append(load_npz(f"{cv_dir}/y_sparse.npz"))

        gene_dates.extend(joblib.load(f"{cv_dir}/gene_list.joblib"))
        taxon.extend(joblib.load(f"{cv_dir}/taxon.joblib"))
        gene_list.extend(joblib.load(f"{cv_dir}/gene_list.joblib"))

    X_sparse = vstack(X_sparse_list)
    y_sparse = vstack(y_sparse_list)

    save_results(X_sparse=X_sparse, y_sparse=y_sparse, gene_list=gene_list, taxon=taxon,
                 gene_dates=gene_dates, feature_list=feature_list, truth_go_list=truth_go_list,
                 output_dir=output_dir, protein_sequences=None)


def save_results_to_row_format(result_path: str, output_path: str):
    def sparse_gt_to_rows(y, protein_ids, go_terms):
        y = y.tocsr()
        rows = []
        for i in range(y.shape[0]):
            start, end = y.indptr[i], y.indptr[i + 1]
            cols = y.indices[start:end]
            for j in cols:
                rows.append(f"{protein_ids[i]}\t{go_terms[j]}")
        return rows

    gt_mat = load_npz(f"{result_path}/y_sparse.npz")
    go_terms = list(joblib.load(f"{result_path}/truth_go_list.joblib"))
    protein_ids = list(joblib.load(f"{result_path}/gene_list.joblib"))

    rows = sparse_gt_to_rows(gt_mat, protein_ids, go_terms)
    with open(output_path, "w") as f:
        f.write("\n".join(rows))


def save_results_information_criteria(row_format_result_path: str, obo_file_path: str, output_path: str):
    annot_file = pd.read_csv(
        row_format_result_path,
        sep="\t", header=None)
    annot_file["aspect"] = "BPO"
    annot_file.columns = ['EntryID', 'term', 'aspect']
    annot_file.to_csv("annot_sequences.txt", sep="\t", index=False)
    ia.run(annotation_path="../annot_sequences.txt", obo_path=obo_file_path, propagate_annotations=True,
           parse_obsolete=True, output_file_path=output_path)
    return output_path


def evaluate_results(evaluation_prediction_dir: str, obo_file_path: str, y_row_format_path: str,
                     information_accretion_path: str, output_dir: str, n_jobs: int):
    # Run the evaluation


    df, dfs_best = cafa_eval(obo_file=obo_file_path, pred_dir=evaluation_prediction_dir, gt_file=y_row_format_path,
                             ia=information_accretion_path, no_orphans=False, norm="cafa", prop="max",
                             max_terms=None, th_step=0.01, n_cpu=n_jobs)

    # Write the results
    write_results(df=df, dfs_best=dfs_best, out_dir=output_dir, th_step=0.01)


def get_key_metrics(result_dir: str) -> Dict[str, float]:
    evaluation_best_f = pd.read_csv(f"{result_dir}/evaluation_best_f.tsv", sep="\t")
    return {"f_max": evaluation_best_f["f"][0]}


def train_and_evaluate_CV_models(optimization_settings:OptimizationSettings, CV_dirs: List[str], obo_file_path: str, training_root_dir: str) -> List[Dict[str, float]]:

    logger.info("Training and evaluating the model for each fold")
    all_cv_evaluations = []
    for test_CV_idx, test_CV_dir in enumerate(CV_dirs):

        logger.info(f"Processing fold {test_CV_idx+1}/{len(CV_dirs)}")
        if test_CV_idx != 4:
            continue
        cv_to_include = [CV_dir for CV_dir in CV_dirs if CV_dir != test_CV_dir]

        cv_train_data_dir = f"{training_root_dir}/training/fold_{test_CV_idx}/training"
        cv_models_dir = (f"{training_root_dir}/training/fold_{test_CV_idx}/models"
                         f"")
        cv_validation_data_dir = f"{training_root_dir}/training/fold_{test_CV_idx}/validation"
        experiment_name = f"train_CV_{test_CV_idx}_{optimization_settings.model_name.value}"
        prediction_result_dir = f"{cv_validation_data_dir}/predictions"
        evaluation_result_dir = f"{cv_validation_data_dir}/evaluations"


        construct_cv_datafile(cv_to_include, cv_train_data_dir)

        shutil.copytree(test_CV_dir, cv_validation_data_dir, dirs_exist_ok=True)

        # Also copy y_sparse to its own file in CAFA format
        y_row_format_path = f"{cv_validation_data_dir}/ground_truth.tsv"
        information_accretion_path = f"{cv_validation_data_dir}/information_accretion.tsv"
        save_results_to_row_format(cv_validation_data_dir, y_row_format_path)
        save_results_information_criteria(row_format_result_path=y_row_format_path,obo_file_path=obo_file_path,output_path=information_accretion_path)



        # Call model training
        logger.info(f"Starting CV {test_CV_idx+1} model for {experiment_name}")
        #
        trained_model_path = train_model(experiment_name=experiment_name, data_dir=cv_train_data_dir, optimization_settings=optimization_settings,
                                         output_model_dir=cv_models_dir)
        trained_model_path = cv_models_dir
        trained_model_path = f"{cv_models_dir}/{experiment_name}_string_search_full_models.joblib"

        # Predict on the evaluation data
        logger.info(f"Predict on evaluation data for {test_CV_idx + 1} model, {experiment_name}")
        evaluation_prediction_dir = predict_with_model(experiment_name=experiment_name, model_path=trained_model_path,
                                                        validation_data_dir=cv_validation_data_dir,
                                                        prediction_result_dir=prediction_result_dir,
                                                        optimization_settings=optimization_settings)

        print(evaluation_prediction_dir)
        logger.info(f"Evaluate predictions for {test_CV_idx+1} model")





        # evaluation_prediction_dir = "/scratch/project_2008455/Max_temp/diamond_search/data/training_data/split_CV/training/fold_4/validation/predictions"

        evaluate_results(evaluation_prediction_dir=evaluation_prediction_dir, obo_file_path=obo_file_path,
                                                y_row_format_path=y_row_format_path, output_dir=evaluation_result_dir,
                         n_jobs=optimization_settings.n_evaluation_jobs,
                         information_accretion_path=information_accretion_path)



        evaluation_results = get_key_metrics(result_dir=evaluation_result_dir)
        all_cv_evaluations.append(evaluation_results)

    return all_cv_evaluations


def train_model(experiment_name: str, data_dir:str, output_model_dir: str,
                optimization_settings:OptimizationSettings):

    go_class_names_path = f"{data_dir}/go_class_names.joblib"
    feature_list_path = f"{data_dir}/feature_list.joblib"
    X_sparse_path = f"{data_dir}/X_sparse.npz"
    y_sparse_path = f"{data_dir}/y_sparse.npz"

    model_trainer = models.ModelTrainer(name=experiment_name, model=getattr(models, optimization_settings.model_name.value),
                                        go_class_names=go_class_names_path, tr_feature_path=X_sparse_path,
                                        number_of_meta_features=optimization_settings.number_of_meta_settings(),
                                        number_of_y_features=optimization_settings.number_of_neighbours,
                                        tr_target_path=y_sparse_path, feature_names=feature_list_path,
                                        return_dummy_if_y_is_uniform=True,
                                        output_path=output_model_dir, **{k:v for k,v in optimization_settings.__dict__.items() if k not in
                                                                         {"model_name", "number_of_hits", "number_of_neighbours", "n_evaluation_jobs", "n_prediction_jobs", "n_training_jobs"}})

    model_trainer.run(n_jobs=optimization_settings.n_training_jobs)

    output_model_path = f"{output_model_dir}/{experiment_name}_string_search_full_models.joblib"
    return output_model_path


def predict_with_model(experiment_name: str, model_path:str, validation_data_dir: str, prediction_result_dir: str,
                       optimization_settings: OptimizationSettings):


    go_class_names = f"{validation_data_dir}/truth_go_list.joblib"
    feature_list_path = f"{validation_data_dir}/feature_list.joblib"
    X_valiation_path = f"{validation_data_dir}/X_sparse.npz"
    gene_list_path = f"{validation_data_dir}/gene_list.joblib"

    predictor = models.Predictor(name=experiment_name, model_path=model_path,
                                 go_class_names=go_class_names, te_feature_path=X_valiation_path,
                                 te_sequences=gene_list_path, output_path=prediction_result_dir,
                                 feature_names=feature_list_path,
                                 number_of_meta_features=optimization_settings.number_of_meta_settings(),
                                 number_of_y_features=optimization_settings.number_of_neighbours,
                                 h5=False,
                                 n_jobs=optimization_settings.n_evaluation_jobs)
    prediction_result_file = predictor.run()

    return str(Path(prediction_result_file).parent)


def hyperparameter_run(optimization_settings: OptimizationSettings,
                       cv_joblib_path: str = CV_JOBLIB_PATH,
                       obo_file_path: str = OBO_FILE_PATH,
                       ground_truth_joblib_path: str = GROUND_TRUTH_PATH):

    CV_dirs = construct_data(optimization_settings=optimization_settings, obo_path=obo_file_path, cv_joblib_path=cv_joblib_path,
                             ground_truth_joblib_path=ground_truth_joblib_path)

    CV_dirs = ['/scratch/project_2008455/Max_temp/diamond_search/data/training_data/split_CV/CV_old_data_CV_split_2', '/scratch/project_2008455/Max_temp/diamond_search/data/training_data/split_CV/CV_old_data_CV_split_4', '/scratch/project_2008455/Max_temp/diamond_search/data/training_data/split_CV/CV_old_data_CV_split_0', '/scratch/project_2008455/Max_temp/diamond_search/data/training_data/split_CV/CV_old_data_CV_split_3', '/scratch/project_2008455/Max_temp/diamond_search/data/training_data/split_CV/CV_old_data_CV_split_1']
    CV_root_dir = Path(CV_dirs[0]).parent

    metrics = train_and_evaluate_CV_models(optimization_settings=optimization_settings, CV_dirs=CV_dirs, obo_file_path=obo_file_path,
                                           training_root_dir=str(CV_root_dir))

    return np.mean([cv_run["f_max"] for cv_run in metrics])


if __name__ == "__main__":

    # target_model = "xgb_train"
    target_model = "lasso_train"
    optimization_settings = OptimizationSettings(number_of_hits=4, number_of_neighbours=2, n_training_jobs=32,
                                                 n_evaluation_jobs=6, n_prediction_jobs=32, model_name=ModelName(target_model))
    k = hyperparameter_run(optimization_settings=optimization_settings)
    print(k)



# ----
# Take the last CV split to val and others to train and perform hyperparameter tuning.
# Hyperparameters:
#   STRING settings
#   Target model type
#   Model parameters
#   ...
# ----

# How do we also create an ensemble of models? Should we combine the top k models for this?




# --------
# - Select the best model based on validation
# --------


# Model testing
# - Run the best model on test to get actual performance metrics