import networkx as nx
import numpy as np
import scipy.sparse
import pickle


def load_Yelp2_data(prefix):

    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()
    features_2 = scipy.sparse.load_npz(prefix + '/features_2.npz').toarray()
    features_3 = scipy.sparse.load_npz(prefix + '/features_3.npz').toarray()
    features_4 = scipy.sparse.load_npz(prefix + '/features_4.npz').toarray()

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    subgraphs = (np.load(prefix + '/subgraphs.npy')).astype(int)
    subgraphs_train_val_test = np.load(prefix + '/subgraphs_train_val_test.npz')

    return [features_0, features_1, features_2, features_3, features_4], \
            subgraphs, \
            subgraphs_train_val_test, \
            labels, \
            adjM, \
            type_mask, \


def load_Freebase_data(prefix):

    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()
    features_2 = scipy.sparse.load_npz(prefix + '/features_2.npz').toarray()
    features_3 = scipy.sparse.load_npz(prefix + '/features_3.npz').toarray()    

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    subgraphs = (np.load(prefix + '/subgraphs.npy')).astype(int)
    subgraphs_train_val_test = np.load(prefix + '/subgraphs_train_val_test.npz')

    return [features_0, features_1, features_2, features_3], \
            subgraphs, \
            subgraphs_train_val_test, \
            labels, \
            adjM, \
            type_mask, \

def load_AminerS_data(prefix):
    
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()
    features_2 = scipy.sparse.load_npz(prefix + '/features_2.npz').toarray()

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    subgraphs = (np.load(prefix + '/subgraphs.npy')).astype(int)
    subgraphs_train_val_test = np.load(prefix + '/subgraphs_train_val_test.npz')

    return [features_0, features_1, features_2], \
            subgraphs, \
            subgraphs_train_val_test, \
            labels, \
            adjM, \
            type_mask, \
            
            
def load_IMDB_data(prefix):

    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()
    features_2 = scipy.sparse.load_npz(prefix + '/features_2.npz').toarray()

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    subgraphs = (np.load(prefix + '/subgraphs.npy')).astype(int)
    subgraphs_train_val_test = np.load(prefix + '/subgraphs_train_val_test.npz')

    return [features_0, features_1, features_2], \
            subgraphs, \
            subgraphs_train_val_test, \
            labels, \
            adjM, \
            type_mask, \


def load_DBLP_data2(prefix):

    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()
    features_2 = np.load(prefix + '/features_2.npy')
    features_3 = np.eye(20, dtype=np.float32)  # one-hot, 20 conf in total

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    subgraphs = (np.load(prefix + '/subgraphs.npy')).astype(int)
    subgraphs_train_val_test = np.load(prefix + '/subgraphs_train_val_test.npz')

    return [features_0, features_1, features_2, features_3], \
            subgraphs, \
            subgraphs_train_val_test, \
            labels, \
            adjM, \
            type_mask


def load_glove_vectors(dim=50):
    print('Loading GloVe pretrained word vectors')
    file_paths = {
        50: 'data/glove.6B.50d.txt',
        100: 'data/glove.6B.100d.txt',
        200: 'data/glove.6B.200d.txt',
        300: 'data/glove.6B.300d.txt'
    }
    f = open(file_paths[dim], 'r', encoding='utf-8')
    wordvecs = {}
    for line in f.readlines():
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        wordvecs[word] = embedding
    print('Done.', len(wordvecs), 'words loaded!')
    return wordvecs
