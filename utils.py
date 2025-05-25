import numpy as np
import pickle
import os

CIFAR10_LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck']

def load_cifar10_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    X = dict[b'data']
    y = dict[b'labels']
    return X, y

def load_cifar10_data(data_dir, classes=[0, 1, 2]):
    X_train, y_train = [], []
    for i in range(1, 6):
        X, y = load_cifar10_batch(os.path.join(data_dir, f'data_batch_{i}'))
        X_train.append(X)
        y_train += y
    X_train = np.concatenate(X_train)
    y_train = np.array(y_train)
    X_test, y_test = load_cifar10_batch(os.path.join(data_dir, 'test_batch'))
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    train_mask = np.isin(y_train, classes)
    test_mask = np.isin(y_test, classes)
    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_test, y_test = X_test[test_mask], y_test[test_mask]
    
    class_map = {c: i for i, c in enumerate(classes)}
    y_train = np.vectorize(class_map.get)(y_train)
    y_test = np.vectorize(class_map.get)(y_test)
    
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    
    y_train_oh = np.eye(len(classes))[y_train]
    y_test_oh = np.eye(len(classes))[y_test]
    return X_train, y_train_oh, X_test, y_test_oh, y_test

