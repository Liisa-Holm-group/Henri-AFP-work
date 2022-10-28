ontology='CC'

# cafa3:

sequences="/home/biotek-groups2/holm/henri/cafa3_data/datasets/${ontology}_sequences.joblib"
features="/home/biotek-groups2/holm/henri/cafa3_data/datasets/${ontology}_cafa3_features.npz"
feature_names="/home/biotek-groups2/holm/henri/cafa3_data/datasets/${ontology}_cafa3_feature_names.joblib"
fasta="/home/biotek-groups2/holm/henri/cafa3_data/cafa3.fasta"
output_path="/home/biotek-groups2/holm/henri/cafa3_data/datasets/"
ipscan="/home/biotek-groups2/holm/henri/cafa3_data/ipr_dir/"
new_cc=0
python generate_stacking_data.py $sequences $features ${feature_names} ${fasta} ${ipscan} $output_path $new_cc



# # cafa3_eval:
# sequences="/home/biotek-groups2/holm/henri/cafa3_data/CAFA3_Features_EvalSet/new_datasets/CAFA_${ontology}_sequences.joblib"
# features="/home/biotek-groups2/holm/henri/cafa3_data/CAFA3_Features_EvalSet/new_datasets/CAFA_${ontology}_cafa3_eval_data_features.npz"
# feature_names="/home/biotek-groups2/holm/henri/cafa3_data/CAFA3_Features_EvalSet/new_datasets/CAFA_${ontology}_cafa3_eval_data_feature_names.joblib"
# fasta="/home/biotek-groups2/holm/henri/cafa3_data/CAFA3_Features_EvalSet/datasets/CAFA3_targets.fasta"
# ipscan="/home/biotek-groups2/holm/henri/cafa3_data/CAFA3_Features_EvalSet/ipr_dir/"
# output_path="/home/biotek-groups2/holm/henri/cafa3_data/new_results/"
# new_cc=0
# python generate_stacking_data.py $sequences $features ${feature_names} ${fasta} ${ipscan} $output_path $new_cc

# new_cc_cafa3:
# sequences="/data/henri/combined_cafa3_cc/new/${ontology}_sequences.joblib"
# features="/data/henri/combined_cafa3_cc/${ontology}_cafa3_features.npz"
# feature_names="/data/henri/combined_cafa3_cc/${ontology}_cafa3_feature_names.joblib"
# fasta="/data/henri/CC_new_CAFA3_data/cafa3.fasta"
# ipscan="/data/henri/CC_new_CAFA3_data/ipr_dir/"
# output_path="/home/biotek-groups2/holm/henri/new_cc_cv_results/"
# new_cc=1
# python generate_stacking_data.py $sequences $features ${feature_names} ${fasta} ${ipscan} $output_path $new_cc

# # in-house data:
# sequences="/data/henri/datasets/${ontology}_sequences.joblib"
# features="/data/henri/datasets/${ontology}_features.npz"
# feature_names="/data/henri/datasets/${ontology}_feature_names.joblib"
# fasta="/data/henri/all.fasta"
# ipscan_cover="/data/henri/ipr_dir/"
# output_path="/data/henri/datasets/"
# new_cc=0
# python generate_stacking_data.py $sequences $features ${feature_names} ${fasta} ${ipscan} $output_path $new_cc
