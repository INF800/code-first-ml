# Make sure:
# IDS col is added before saving 

import argparse, textwrap
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import model_selection


# =======================================================================================================================================
# beg: kfold for classification dataset
# =======================================================================================================================================
def create_classification_folds(data, k, shuffled=False):
    """data is df for regression dataset
    
    :param data: pandas df with format mentioned below
    :param k: k in f-fold
    :pram shuffled: Returns shuffled df with `k` if True else unshuffled

    Note: \
        The fomat of input `data` df must be as follows
        + All columns must be named either `f_n` or `target`
        + All columns must be cleaned and preprocessed i.e 
            - Numbers (int / float)
            - No nulls
    """

    data["kfold"] = -1
    data = data.sample(frac=1).reset_index(drop=True)
    kf = model_selection.StratifiedKFold(n_splits=k)
    y = df.target.values

    # fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)): 
        df.loc[v_, 'kfold'] = f

    # return dataframe with folds
    # shuffle if requiredrue)
    if shuffled is True : return data.sample(frac=1).reset_index(drop=True)
    else: return data
# =======================================================================================================================================
# beg: kfold for classification dataset
# =======================================================================================================================================



# =======================================================================================================================================
# beg: kfold for regression dataset
# =======================================================================================================================================
def create_regression_folds(data, k, shuffled=False):
    """data is df for regression dataset
    
    :param data: pandas df with format mentioned below
    :param k: k in f-fold
    :pram shuffled: Returns shuffled df with `k` if True else unshuffled

    Note: \
        The fomat of input `data` df must be as follows
        + All columns must be named either `f_n` or `target`
        + All columns must be cleaned and preprocessed i.e 
            - Numbers (int / float)
            - No nulls
    """

    data["kfold"] = -1
    data = data.sample(frac=1).reset_index(drop=True)
    
    # bin targets
    num_bins = int(np.floor(1 + np.log2(len(data))))
    data.loc[:, "bins"] = pd.cut(
        data["target"], bins=num_bins, labels=False
    )

    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=k)

    # fill the new kfold column
    # note that, instead of targets, we use bins as if targets were for classification!
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'kfold'] = f

    # drop the bins column and
    # return dataframe with folds
    data = data.drop("bins", axis=1)
    if shuffled is True : return data.sample(frac=1).reset_index(drop=True)
    else: return data
# =======================================================================================================================================
# end: kfold for regression dataset
# =======================================================================================================================================



# =======================================================================================================================================
# beg: arg parser
# =======================================================================================================================================
ap = argparse.ArgumentParser(
      prog='MakeKFold',
      description="Make kfold column for preprocessed classification / regression dataset",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=textwrap.dedent('''\
        Usage: 
        Goto main directory and run
            python src/make_kfold.py -k 5 -t regression -s true -i false -f <name-of-preprocessed-csv-file-in-input-dir>
        Examples:
            python src/make_kfold.py -k 5 -t regression -s true -i true -f preprocessed_train_regression.csv
        Requirements:
            python 3.7+
         '''))

ap.add_argument("-k", "--k", required=True, type=int, help="k in kfold")
ap.add_argument("-t", "--type-of-problem", required=True, type=str, help="classification or regression")
ap.add_argument("-s", "--return-shuffled", required=True, type=str, help="Time seies False, else True")
ap.add_argument("-i", "--add-id-col", required=True, type=str, help="Returns df with IDS col for merging back if True")
ap.add_argument("-f", "--preprocessed-file", required=True, type=str, help="Name of preprocessed csv file in input dir")

args = vars(ap.parse_args())
# print("DEBUG", args)
# =======================================================================================================================================
# end: arg parser
# =======================================================================================================================================



if __name__ == "__main__":
    str2bool = lambda x:  True if x in ['true', 'True', 1] else False

    k = args['k']
    t = args['type_of_problem']
    s = args['return_shuffled'] # str. Not bool.
    i = args['add_id_col'] # str. Not bool.
    f = args['preprocessed_file']

    df = pd.read_csv(f'input/{f}')
    s = str2bool(s)
    i = str2bool(i)

    # print(f"s is {s}")
    # print(f"i is {i}")

    if t == 'classification':
        df = create_classification_folds(df, k=k, shuffled=s)
    elif t == 'regression':
        df = create_regression_folds(df, k=k, shuffled=s)
    else:
        raise Exception('Type must be either regression or classification')
    
    # print("DEBUG ", df.skew())

    
    """
    save to csv
    """

    df["IDS"] = list(range(len(df)))
    # print("DEBUG ", df.kfold.value_counts())
    df.to_csv("input/train_folds.csv", index=False)