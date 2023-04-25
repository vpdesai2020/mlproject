import os
import sys

import numpy as np 
import pandas as pd
# Import required libraries for model training and evaluation
import dill # used to serialize and deserialize Python objects
import pickle # used to save and load Python objects
from sklearn.metrics import r2_score # used to evaluate the performance of the regression model
from sklearn.model_selection import GridSearchCV # used for hyperparameter tuning using grid search


from src.exception import CustomException

def save_object(file_path, obj):
    '''
    This function takes the file path and object as input and saves the object into the file using pickle.
    It also creates directories for the file path if they don't exist.
    
    Args:
    file_path: path where object is to be saved
    obj: object to be saved
    
    Returns:
    None
    '''

    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)