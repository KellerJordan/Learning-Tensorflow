import tempfile
from six.moves import urllib
train_file = tempfile.NamedTemporaryFile(delete=False)
test_file = tempfile.NamedTemporaryFile(delete=False)
urllib.request.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/voting-records/house-votes-84.data", train_file.name)
urllib.request.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/voting-records/house-votes-84.data", test_file.name)

import pandas as pd
COLUMNS = ["political-affiliation",
        "handicapped-infants", "water-project-cost-sharing", "adoption-of-the-budget-resolution",
        "physician-fee-freeze", "el-salvador-aid", "religious-groups-in-schools", "anti-satellite-test-ban",
        "aid-to-nicaraguan-contras", "mx-missile", "mx-missile", "synfuels-corporation-cutback", "education-spending", 
        "superfund-right-to-sue", "crime", "duty-free-exports", "export-administration-act-south-africa"]
df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)

LABEL_COLUMN = "label"
df_train[LABEL_COLUMN] = (df_train["political-affiliation"].apply(lambda x: 1 if x == 'democrat' else 0))
df_test[LABEL_COLUMN] = (df_test["political-affiliation"].apply(lambda x: 1 if x == 'democrat' else 0))
# df_train[LABEL_COLUMN] = (df_train["crime"].apply(lambda x: 1 if x == 'y' else 0))
# df_test[LABEL_COLUMN] = (df_test["crime"].apply(lambda x: 1 if x == 'y' else 0))

CATEGORICAL_COLUMNS = [
        # "political-affiliation",
        "handicapped-infants", "water-project-cost-sharing", "adoption-of-the-budget-resolution",
        "physician-fee-freeze", "el-salvador-aid", "religious-groups-in-schools", "anti-satellite-test-ban",
        "aid-to-nicaraguan-contras", "mx-missile", "mx-missile", "synfuels-corporation-cutback", "education-spending", 
        "superfund-right-to-sue", "crime", "duty-free-exports", "export-administration-act-south-africa"]

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

def input_fn(df):
    feature_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values,
        dense_shape=[df[k].size, 1]
    ) for k in CATEGORICAL_COLUMNS}
    label = tf.constant(df[LABEL_COLUMN].values)
    return feature_cols, label

def train_input_fn():
    return input_fn(df_train)

def eval_input_fn():
    return input_fn(df_test)

feature_columns = [
    # tf.contrib.layers.sparse_column_with_keys(column_name="political-affiliation", keys=["democrat", "republican"]),

    tf.contrib.layers.sparse_column_with_keys(column_name="handicapped-infants", keys=["y", "n"]),
    tf.contrib.layers.sparse_column_with_keys(column_name="water-project-cost-sharing", keys=["y", "n"]),
    tf.contrib.layers.sparse_column_with_keys(column_name="adoption-of-the-budget-resolution", keys=["y", "n"]),
    tf.contrib.layers.sparse_column_with_keys(column_name="physician-fee-freeze", keys=["y", "n"]),
    tf.contrib.layers.sparse_column_with_keys(column_name="el-salvador-aid", keys=["y", "n"]),
    tf.contrib.layers.sparse_column_with_keys(column_name="religious-groups-in-schools", keys=["y", "n"]),
    tf.contrib.layers.sparse_column_with_keys(column_name="anti-satellite-test-ban", keys=["y", "n"]),
    tf.contrib.layers.sparse_column_with_keys(column_name="superfund-right-to-sue", keys=["y", "n"]),
    tf.contrib.layers.sparse_column_with_keys(column_name="crime", keys=["y", "n"]),
    tf.contrib.layers.sparse_column_with_keys(column_name="duty-free-exports", keys=["y", "n"]),
    tf.contrib.layers.sparse_column_with_keys(column_name="export-administration-act-south-africa", keys=["y", "n"]),
]

model_dir = tempfile.mkdtemp()
m = tf.contrib.learn.LinearClassifier(feature_columns=feature_columns, model_dir=model_dir)

m.fit(input_fn=train_input_fn, steps=200)
results = m.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))