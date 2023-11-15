from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
import xgboost
from sklearn.pipeline import Pipeline
import logging, sys

from app.column_transformer import ColumnTransformer

class DataFitting():
    def __init__(self):
        self.rfr_model = joblib.load('models/rfr_model_joblib.joblib')

        self.pipeline = Pipeline(
                steps=[
                    ("convert conditional", ColumnTransformer())
                ]
        )    

        param_grid = [
            {'subsample': [0.5], 'n_estimators': [1400], 
            'max_depth': [5], 'learning_rate': [0.02],
            'colsample_bytree': [0.4], 'colsample_bylevel': [0.5],
            'reg_alpha':[1], 'reg_lambda': [1], 'min_child_weight':[2]}
        ]

        xgb = xgboost.XGBRegressor(eval_metric='rmse')

        self.xgb_grid = GridSearchCV(xgb, param_grid, cv=3, verbose=1, scoring='neg_root_mean_squared_error')

        self.train = pd.read_csv('data/train.csv', index_col='Id')

        # Remove rows with missing target
        self.train = self.train.dropna(axis=0, subset=['SalePrice'])

        #Creating Y and X
        self.X = self.train.drop(columns='SalePrice', axis=1)

        self.y = self.train.SalePrice
        self.X = self.pipeline.fit_transform(self.X)

    def getCategoricalColumns(self):

        self.train = self.train.drop(columns="Utilities", axis=1)
        self.train = self.train.drop(columns="Street", axis=1)
        self.train = self.train.drop(columns="MiscFeature", axis=1)
        self.train = self.train.drop(columns="MiscVal", axis=1)

        string_columns = self.train.select_dtypes(include=["object"])
        quality_columns = []

        for i in string_columns:
            if ("GD" in self.train[i].unique()) or ("Ex" in self.train[i].unique()) or ("Po" in self.train[i].unique()) or ("LwQ" in self.train[i].unique()) or ("Reg" in self.train[i].unique()) or ("Unf" in self.train[i].unique()) or ("Typ" in self.train[i].unique()):
                quality_columns.append(i)

        # Some of these categories contain NaN values, we will replace this with a string "NA"
        self.train.loc[:,quality_columns] = self.train.loc[:,quality_columns].fillna('NA')
        

        for column in string_columns:
            self.train[column] = self.train[column].mask(self.train[column].map(self.train[column].value_counts(normalize=True)) < 0.01, 'Other')
        columnsWithUniqueValues = []
        
        
        for i in string_columns:
            
            columnsWithUniqueValues.append([i, self.train[i].unique()])
            
        return columnsWithUniqueValues

    def getIntColumns(self):
        float_columns = self.train.select_dtypes(include=["float64"])
        for i in float_columns:
            self.train[i] = self.train[i].fillna(0).astype("int64")

        int_columns = self.train.select_dtypes(include=["int64"])
    
        return int_columns.columns


    def predictXGB(self, data):
        
        data = self.pipeline.fit_transform(data)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2 , random_state=1)
        
        self.xgb_grid.fit(X_train, y_train)

        pred = self.xgb_grid.predict(data)

        y_pred = self.xgb_grid.predict(X_test)

        score = r2_score(y_pred, y_test, multioutput='variance_weighted')
        
        return pred, score