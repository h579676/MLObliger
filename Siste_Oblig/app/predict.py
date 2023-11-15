import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
import xgboost
from sklearn.pipeline import Pipeline
import locale 

from app.column_transformer import ColumnTransformer
from app.data_fitting import DataFitting

####### 
## Get the model trained in the notebook 
# `../nbs/1.0-asl-train_model.ipynb`
#######

model = DataFitting()




#xgb_grid.fit(X_train, y_train)


def preprocess(data):
    """
    
Retrieves user-entered web form features with default values for 
simplicity. Defaults are set based on common values for booleans and integers, 
and the median for floats. Note that these assumptions may not apply universally, 
and careful consideration is essential. Emphasizes the importance of dynamic defaults and 
suggests obtaining features when possible, either from user profiles or other available means.
    """
    train = pd.read_csv('data/train.csv', index_col='Id')
   

    feature_values = { 
        'MSSubClass': 60, 
        'MSZoning' : 3, 
        'LotFrontage' : 65.0, 
        'LotArea' : 8450, 
        'Alley': -1,
        'LotShape': 0, 
       'LandContour': 3, 
       'LotConfig': 3, 
       'LandSlope': 0, 
       'Neighborhood': 4,
       'Condition1': 2, 
       'Condition2': 0, 
       'BldgType': 0, 
       'HouseStyle': 2, 
       'OverallQual': 7,
       'OverallCond':5, 
       'YearBuilt':2003, 
       'YearRemodAdd':2003, 
       'RoofStyle':0, 
       'RoofMatl':0,
       'Exterior1st':8, 
       'Exterior2nd':8, 
       'MasVnrType':1,
        'MasVnrArea':196.0, 
        'ExterQual':3,
       'ExterCond':2, 
       'Foundation':3, 
       'BsmtQual':3, 
       'BsmtCond':3, 
       'BsmtExposure':3,
       'BsmtFinType1':6, 
       'BsmtFinSF1':706, 
       'BsmtFinType2':1, 
       'BsmtFinSF2':0, 
       'BsmtUnfSF':150,
       'TotalBsmtSF':186, 
       'Heating':0, 
       'HeatingQC':4, 
       'CentralAir':1, 
       'Electrical':3,
       '1stFlrSF':856, 
       '2ndFlrSF':854, 
       'LowQualFinSF': 0, 
       'GrLivArea': 1710, 
       'BsmtFullBath': 1,
       'BsmtHalfBath':0, 
       'FullBath':2, 
       'HalfBath':1, 
       'BedroomAbvGr':3, 
       'KitchenAbvGr':1,
       'KitchenQual':3, 
       'TotRmsAbvGrd':0, 
       'Functional':8, 
       'Fireplaces':0,
       'FireplaceQu':0, 
       'GarageType':0, 
       'GarageYrBlt':2003.0, 
       'GarageFinish':2,
       'GarageCars':2, 
       'GarageArea':548, 
       'GarageQual':3, 
       'GarageCond':3, 
       'PavedDrive':2,
       'WoodDeckSF':0, 
       'OpenPorchSF':61, 
       'EnclosedPorch':0, 
       '3SsnPorch':0,
       'ScreenPorch':0, 
       'PoolArea':0, 
       'PoolQC':0, 
       'Fence':-1,  
       'MoSold':2, 
       'YrSold':2008, 
       'SaleType':3, 
       'SaleCondition':2
    }
    train = train.drop(columns='SalePrice', axis=1)
    feature_values = train.sample()

    # Parse the form inputs and return the defaults updated with values entered.
    
    for key in [k for k in data.keys() if k in feature_values.keys()]:
        feature_values[key] = data[key]
    
    return feature_values



####### 
## predict with the trained model:
#######


def predict(data):
    """
    If debug, print various useful info to the terminal.
    """
 
    # Store the data in an array in the correct order:

    column_order = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Alley', 'LotShape',
       'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
       'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond',
       'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond',
       'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
       'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF',
       '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
       'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition']

    #data = np.array([data[feature] for feature in column_order]).reshape(1,-1)
    #data = pd.DataFrame(data)

    # NB: In this case we didn't do any preprocessing of the data before 
    # training our random forest model (see the notebool `nbs/1.0-asl-train_model.ipynb`). 
    # If you plan to feed the training data through a preprocessing pipeline in your 
    # own work, make sure you do the same to the data entered by the user before 
    # predicting with the trained model. This can be achieved by saving an entire 
    # sckikit-learn pipeline, for example using joblib as in the notebook.
    #data.columns = column_order
    print(data)
    pred, score = model.predictXGB(data)

    #uncertainty = model.predict_proba(data.reshape(1,-1))
    
    return pred, score


def postprocess(prediction):
    """
    Apply postprocessing to the prediction. E.g. validate the output value, add
    additional information etc. 
    """

    pred, score = prediction

    # Validate. As an example, if the output is an int, check that it is positive.
    try: 
        int(pred[0]) > 0
    except:
        pass

    # Make strings
    pred = str(pred[0])
    
    #score = round(score, 1) * 100

    score = f"{score*100:.2f} %"

    locale.setlocale(locale.LC_ALL, 'en-US')
    
    pred = locale.currency(int(float(pred)), grouping=True)
    # Return
    return_dict = {'pred': pred, "score": score}

    return return_dict