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
    

def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)