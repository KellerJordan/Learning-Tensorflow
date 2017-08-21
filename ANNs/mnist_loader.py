import pickle
import gzip
import numpy as np

def load_data():
    with gzip.open('../data/mnist.pkl.gz', 'rb') as f:
        tr_d, va_d, te_d = pickle.load(f, encoding='latin1')
    training_data = format_data(tr_d, vectorize=True)
    validation_data = format_data(va_d)
    test_data = format_data(te_d)
    return training_data, validation_data, test_data

def format_data(data, vectorize=False):
    def reshape_features(features):
        return [np.reshape(x, (784,)) for x in features]
    def vectorize_labels(labels):
        def vectorize_label(label):
            one_hot = np.zeros((10,))
            one_hot[label] = 1.0
            return one_hot
        return [vectorize_label(y) for y in labels]
    data_features = reshape_features(data[0])
    data_labels = vectorize_labels(data[1]) if vectorize else data[1]
    return list(zip(data_features, data_labels))
