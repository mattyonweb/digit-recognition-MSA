from typing import *
import numpy as np

def __merge_h5_train_test_sets(xtr: np.array, ytr, xte, yte):
    return (
        np.concatenate((xtr, xte)),
        np.concatenate((ytr, yte))
    )


def randomize(X: np.array, Y: np.array) -> Tuple[np.array, np.array]:
    permutation = np.random.permutation(len(X))
    
    newx = X[permutation]
    newy = Y[permutation]
    
    return newx, newy

    
def generate_folds(X, Y, k):
    X, Y = randomize(X, Y)
    
    test_size = len(Y) // k
    
    for i in range(k):
        assert len(X) == len(Y)
        
        testX, trainX = X[:test_size], X[test_size:]
        testY, trainY = Y[:test_size], Y[test_size:]
        
        yield (trainX, trainY), (testX, testY)

        X, Y = np.roll(X, test_size, axis=0), np.roll(Y, test_size, axis=0)
        
# xall = np.array([ np.array([_] * 14) for _ in range(23)])
# yall = np.array([1 for _ in range(23)])

