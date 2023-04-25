# Import required modules
import sys
import os
from dataclasses import dataclass
import numpy as np 
import pandas as pd
# Import required libraries for data transformation
from sklearn.compose import ColumnTransformer # used to apply different transformations to different columns of data
from sklearn.impute import SimpleImputer # used to impute missing values in the data
from sklearn.pipeline import Pipeline # used to create a pipeline of transformations to be applied to the data
from sklearn.preprocessing import OneHotEncoder,StandardScaler # used for one-hot encoding and standardization of the data


# Import custom modules
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


#  Define dataclass for configuration related to data transformation
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")


# Define DataTransformation class to perform data transformation
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function defines the data transformation pipeline for numerical and 
        categorical columns separately using `sklearn.pipeline.Pipeline` and `sklearn.compose.ColumnTransformer`. 
        The numerical pipeline contains steps for SimpleImputer and StandardScaler, 
        while the categorical pipeline contains steps for SimpleImputer, OneHotEncoder, and StandardScaler. 
        Finally, it returns a preprocessor object.
        
        Args:
        None
        
        Returns:
        preprocessor: a preprocessor object for data transformation
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        '''
        This function reads the train and test data from CSV files, 
        applies the preprocessor object on the input features of train and test data, 
        and returns the transformed train and test data along with the path where preprocessor object is saved.
        
        Args:
        train_path: file path of the train data CSV file
        test_path: file path of the test data CSV file
        
        Returns:
        train_arr: transformed train data
        test_arr: transformed test data
        preprocessor_file_path: file path where preprocessor object is saved
        '''
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
