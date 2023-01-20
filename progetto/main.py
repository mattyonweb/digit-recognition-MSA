from typing import *

import h5py
import random
import time
import pickle
import numpy as np
from numpy.linalg import norm

import progetto.kfold as kfold

    
def kernel_poly(x1, x2, exp=2):
    return (1 + x1.dot(x2)) ** exp

def kernel_gauss(x1, x2, sigma=2):
    return np.exp(-np.linalg.norm(x1 - x2)**2 / (2 * sigma**2))

def sign(x):
    return 1 if x >= 0 else -1


# =========================================================== #
# =========================================================== #
# Preprocessing

def import_dataset() -> Tuple:
    with h5py.File("data/usps.h5", 'r') as hf:
        train = hf.get('train')
        X_tr = train.get('data')[:]
        y_tr = train.get('target')[:]
        test = hf.get('test')
        X_te = test.get('data')[:]
        y_te = test.get('target')[:]

    # Data already normalized
    assert max(np.max(X_tr), np.max(X_te)) == 1
    assert min(np.min(X_tr), np.min(X_te)) == 0

    # Meaningless values in labels? (just a sanity check)
    for i, y in enumerate(y_tr):
        assert y in [0,1,2,3,4,5,6,7,8,9]
    for i, y in enumerate(y_te):
        assert y in [0,1,2,3,4,5,6,7,8,9]

    all_X, all_Y = kfold.__merge_h5_train_test_sets(X_tr, y_tr, X_te, y_te)

    return all_X, all_Y   


# =========================================================== #
# =========================================================== #
# Training algorithm

def pegasos(X, Y, for_digit, kernel, lambd, T) -> np.array:
    """
    Train a predictor for a given digit.
    """
    alpha = []

    for t in range(1, T+1):
        idx = random.randint(0, len(X)-1)
        x, y = X[idx], Y[idx]

        if t%1000 == 0:
            print(t, len(alpha))
            
        y = 1 if y == for_digit else -1

        prediction  = (1/(lambd*t))
        prediction *= sum(ys * kernel(xs, x) for xs,ys in alpha)

        if y * prediction < 1:
            alpha.append((x,y))

    return alpha
    # return lambda x: (
    #     sum(ys * kernel(xs, x) for xs,ys in alpha)
    # )


def pegasos_predictors(X, Y, lambd, T, kernel) -> (List[Tuple[list,Callable]], float):
    """
    Train predictors for all 10 digits.

    Also returns the time required for each digit training.
    """
    predictors = list()
    times      = list()
    
    for digit in range(0, 10):
        print(f"Building predictor for digit {digit}")

        start = time.time()
        predictor_alpha = pegasos(X,Y,digit,kernel,lambd,T)
        predictor_func  = callable_predictor(predictor_alpha, kernel)
        end   = time.time() - start
        
        predictors.append((predictor_alpha, predictor_func))
        times.append(end)
        
    return (predictors, sum(times) / 10)

def callable_predictor(alpha: list, kernel: Callable) -> Callable:
    return lambda x: (
        sum(ys * kernel(xs, x) for xs,ys in alpha)
    )

# =========================================================== #
# =========================================================== #
# Test error calculation

def test_error(X, Y, predictors: List[Callable]) -> float:
    errors = 0

    print("Testing...")
    
    for t, (x,y) in enumerate(zip(X,Y)):
        if t!=0 and t%500 == 0:
            print(t)

        predictions = [predictor(x) for predictor in predictors]
        predictions = np.array(predictions)

        best_num = np.argmax(predictions)

        if best_num != y:
            errors += 1

    print(f"Test error: {errors/t}")
    return errors/t

# =========================================================== #
# =========================================================== #


def single_train_and_test(X, Y, lambd, T, kernel) -> Tuple[Tuple[list, Callable], float, int, float]:
    """
    Performs an estimation of test error using 5-fold cross-validation.
    """
    print(f"Running on: lambda={lambd} and T={T}")
    
    test_errors = list()
    
    single_round_times = list()
    single_digit_train_times = list()

    best_testerror_sofar:  float = float("inf")
    best_predictors_sofar: Tuple[list, Callable] = None
    
    for (xtr, ytr), (xte, yte) in kfold.generate_folds(X, Y, 5):
        start = time.time()

        # Build predictors for all digits
        predictors, single_digits_avg_train_time = pegasos_predictors(xtr, ytr, lambd, T, kernel)

        print(f"Predictors generation took: {int(time.time() - start)} seconds")
        single_digit_train_times.append(
            single_digits_avg_train_time
        )

        # calculate test error directly with callable predictor
        error = test_error(xte, yte, [p[1] for p in predictors])
        test_errors.append(error)
        if error < best_testerror_sofar:
            best_testerror_sofar  = error
            best_predictors_sofar = predictors

        
        single_round_times.append(int(time.time() - start))
        
        print(f"Single round test-error for lambda={lambd}, T={T}: " +
              f"{test_errors[-1]} in {single_round_times[-1]} seconds")


    avg_single_digit_train_time = int(sum(single_digit_train_times) / len(single_digit_train_times))
    avg_test_error              = sum(test_errors) / len(test_errors)
    avg_round_time              = int(sum(single_round_times)/len(single_round_times))
    
    print("Completed 5-fold train and test")
    print(f"Average single digit train time: {avg_single_digit_train_time}")
    print(f"Average test error: {avg_test_error}")
    print(f"Average duration of single round: {avg_round_time} seconds")
    print(f"Total time: {sum(single_round_times)} seconds")
    
    return (best_predictors_sofar, avg_test_error, avg_single_digit_train_time, avg_round_time)


# =========================================================== #
# =========================================================== #


if __name__ == "__main__":
    all_X, all_Y = import_dataset()
    
    # kernel function hyperparameter
    kernels = [
        (lambda x,y: kernel_poly(x,y,1), "poly exp=1"),
        (lambda x,y: kernel_poly(x,y,3), "poly exp=3"),
        (lambda x,y: kernel_poly(x,y,7), "poly exp=7"),
        (lambda x,y: kernel_gauss(x,y,sigma=2), "gauss sigma=2"),
    ]

    # Num of epochs hyperparameter
    __trainset_size = 4 * (len(all_X) // 5)
    training_sizes = [
        int(x * __trainset_size) for x in [0.1, 0.5, 1, 2]
    ]

    # Lambda hyperparameter
    lambdas = [10**-8, 10**-7, 10**-6, 10**-5]


    results   = list()
    last_time = time.time()


    for kernel, kernel_name in kernels:
        for T in training_sizes:
            for lambd in lambdas:
                start_time = time.time()

                predictors, avg_test_error, digit_train_time, round_time = (
                    single_train_and_test(all_X, all_Y, lambd, T, kernel)
                )

                with open("results.csv", "a") as f:
                    f.write(f"{kernel_name},{lambd},{T},{avg_test_error},{digit_train_time},{round_time}\n")

                # with open(f"results/predictors/{kernel_name}_{T}_{lambd}.bin", "wb") as f:
                #     pickle.dump(
                #         [digit_pred[0] for digit_pred in predictors], f
                #     )

                print(f"Total time: {time.time() - start_time} seconds\n\n")

    print(f"Total time: {time.time() - last_time}")





