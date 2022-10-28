import argparse
import sys
import joblib
import scipy.sparse as sp
from Bio import SeqIO
from pdb import set_trace as bp
from sklearn.feature_extraction import DictVectorizer
import load_data 
import data_processing
import numpy as np

def seq_features(fname, new_cc, cafa3_eval=False):
    res = {}
    for seq in SeqIO.parse(fname, 'fasta') :
        SeqInput = str(seq.seq)
        if not cafa3_eval:
            name = str(seq.id).split('|')[1]
        else:
            if new_cc:
                name = str(seq.id).split('|')[1]
            else:
                name = str(seq.id)
        lenght = len(seq.seq)
        NoiseChars = "UXOJZB"
        RemoveCount = 0
        for k in range(len(NoiseChars)):
            RemoveCount = RemoveCount + SeqInput.count(NoiseChars[k])
        res[name] = {'x':RemoveCount/lenght, 'lenght':lenght}
    return res

def generate_data(sequences, features, feature_names, fasta, ipscan_path, output_path, new_cc):

    sf = seq_features(fasta, new_cc)
    ipscan_cover = data_processing.ipscan_cover(ipscan_path)

    # sum(sf[seq]['x'] > 0 for seq in sf.keys()) # NOTE 8093/897186 contain noise chars

    sequences = joblib.load(sequences)
    features = sp.load_npz(features).tocsr()
    feature_names = joblib.load(feature_names)

    # read sequence features
    vectorizer = DictVectorizer()
    values = vectorizer.fit_transform(load_data.transform_features(sequences, sf, dict))
    seq_feature_names = vectorizer.get_feature_names()

    # read ipscan features
    ipscan_index = [i for i, n in enumerate(feature_names) if 'cluster' in n or 'e_value' in n]
    ipscan_e_value_index = [i for i, n in enumerate(feature_names) if 'e_value' in n]
    ipscan_counts = np.count_nonzero(features[:,ipscan_index].toarray(), axis=1)
    max_ipscan = features[:,ipscan_e_value_index].toarray().max(axis=1)

    cover = []
    for s in sequences:
        try:
            cover.append(ipscan_cover[s])
        except KeyError:
            cover.append(0.1)

    t = values[:,0].toarray().squeeze()
    cover = np.array(cover) / np.where(t>0, t ,1)

    # combine features
    res = np.hstack((values.toarray(), ipscan_counts.reshape(-1,1), cover.reshape(-1,1), max_ipscan.reshape(-1,1)))
    names = seq_feature_names + ['ipscan_count', 'ipscan_cover', 'max_ipscan']

    # taxonomy features
    t = features.tocsr()
    index = [i for i, n in enumerate(feature_names) if 'taxonomy' in n]
    taxonomy_features = t[:,index]

    joblib.dump(names, f'{output_path}/{ontology}_stacking_feature_names.joblib')
    np.save(f'{output_path}/{ontology}_stacking_features.npy', res)
    sp.save_npz(f'{output_path}/{ontology}_stacking_taxonomy.npz', taxonomy_features)


def main():
    """CLI for generating datasets."""

    parser = argparse.ArgumentParser()
    parser.add_argument('sequences',type=str)
    parser.add_argument('features',type=str)
    parser.add_argument('feature_names',type=str)
    parser.add_argument('fasta',type=str, help='fasta file')
    parser.add_argument('ipscan',type=str, help='ipscan directory')
    parser.add_argument('output_path',type=str )
    parser.add_argument('new_cc',type=int, help='new_cc = 1 otherwise 0')
    args = parser.parse_args()

    generate_data(args.sequences, args.features, args.feature_names, args.fasta, args.ipscan, args.output_path, args.new_cc)


if __name__ == '__main__':
    main()


