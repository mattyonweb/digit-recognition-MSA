"""
To test predictors stored in /results/predictors
"""
import h5py

from typing import *
from pickle import load
from progetto.main import kernel_poly, kernel_gauss, test_error, import_dataset, callable_predictor


def select_kernel(predictor_name: str) -> Callable:
    if predictor_name.startswith("poly exp="):
        exp = int(predictor_name[9])
        return lambda x,y: kernel_poly(x,y,exp)
    
    return lambda x,y: kernel_gauss(x,y,sigma=2)

def import_predictors(predictor_name: str) -> List[Callable]:
    """
    Imports the predictors stored in a .bin file; then returns
    10 predictors functions, one for each digit.
    """
    fpath = f"results/predictors/{predictor_name}.bin"

    with open(fpath, "rb") as f:
        # list of 10 alpha lists, one for each digit
        alphas: List[list] = load(f)

    kernel = select_kernel(predictor_name)

    return [callable_predictor(alpha, kernel) for alpha in alphas]


if __name__ == "__main__":
    # change this to test another pre-trained model
    predictor_name = "poly exp=1_743_1e-05"

    predictors = import_predictors(predictor_name)

    with h5py.File("data/usps.h5", 'r') as hf:
        test = hf.get('test')
        X_te = test.get('data')[:]
        y_te = test.get('target')[:]
        
    test_error(X_te, y_te, predictors)
    
        
