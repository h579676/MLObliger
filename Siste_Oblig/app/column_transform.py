from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import pandas as pd
import warnings as w
w.filterwarnings('ignore')

class ColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        
        quality_columns = []

        string_columns = X.select_dtypes(include=["object"])
        
        for i in string_columns:
            if ("GD" in X[i].unique()) or ("Ex" in X[i].unique()) or ("Po" in X[i].unique()) or ("LwQ" in X[i].unique()) or ("Reg" in X[i].unique()) or ("Unf" in X[i].unique()) or ("Typ" in X[i].unique()):
                quality_columns.append(i)

        # Some of these categories contain NaN values, we will replace this with a string "NA"
        X.loc[:,quality_columns] = X.loc[:,quality_columns].fillna('NA')
        
        columns1 = ["FireplaceQu", "GarageQual", "GarageCond"]
        X.loc[:,columns1] = X.loc[:,columns1].replace(['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], [0,1,2,3,4,5])

        columns2 = ["PoolQC"]
        X.loc[:,columns2] = X.loc[:,columns2].replace(['NA', 'Fa', 'Gd', 'Ex'], [0,1,2,3])

        columns3 = ["KitchenQual", "ExterQual"]
        X.loc[:,columns3] = X.loc[:,columns3].replace(['NA', 'Fa', 'TA', 'Gd', 'Ex'], [0,1,2,3,4])

        columns4 = ["HeatingQC", "ExterCond"]
        X.loc[:,columns4] = X.loc[:,columns4].replace(['Po', 'Fa', 'TA', 'Gd', 'Ex'], [0,1,2,3,4])

        columns5 = ["BsmtFinType1", "BsmtFinType2"]
        X.loc[:,columns5] = X.loc[:,columns5].replace(['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], [0,1,2,3,4,5,6])

        columns6 = ["BsmtCond"]
        X.loc[:,columns6] = X.loc[:,columns6].replace(['NA', 'Po', 'Fa', 'TA', 'Gd'], [0,1,2,3,4])

        columns7 = ["BsmtQual"]
        X.loc[:,columns7] = X.loc[:,columns7].replace(['NA', 'Fa', 'TA', 'Gd', 'Ex'], [0,1,2,3,4])
      
        columns8 = ["LotShape"]
        X.loc[:,columns8] = X.loc[:,columns8].replace(['Reg', 'IR1', 'IR2', 'IR3'], [0,1,2,3])
        
        columns9 = ["GarageFinish"]
        X.loc[:,columns9] = X.loc[:,columns9].replace(['NA', 'Unf', 'RFn', 'Fin' ], [0,1,2,3])
       
        columns10 = ["Functional"]
        X.loc[:,columns10] = X.loc[:,columns10].replace(['NA', "Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"], [0,1,2,3,4,5,6,7,8])
        

        columns11 = ["PavedDrive"]
        X.loc[:,columns11] = X.loc[:,columns11].replace(["N", "P", "Y"], [0,1,2])
        
        
        # Dropping columns
        if "Utilities" in X:
            X = X.drop(columns="Utilities", axis=1)
        if "Street" in X:
            X = X.drop(columns="Street", axis=1)
        if "MiscFeature" in X:
            X = X.drop(columns="MiscFeature", axis=1)
        if "MiscVal" in X:
            X = X.drop(columns="MiscVal", axis=1)

        
        # Converting low frequency to other
        columns = number_of_string_columns(X)

        for column in columns:
            X[column] = X[column].mask(X[column].map(X[column].value_counts(normalize=True)) < 0.01, 'Other')
        
        
        # Label encoding
        columns = number_of_string_columns(X)
  
        for i in columns:
            X[i] = pd.Categorical(X[i]).codes
            
         
        
        # fixing nan values
        rmissingvaluecol(X,0.01, 0)

    
        return X
    
    
 
    
    
def rmissingvaluecol(dff,threshold, newValue):
    l = []
    l = list(dff.drop(dff.loc[:,list((100*(dff.isnull().sum()/len(dff.index))>=threshold))].columns, 1).columns.values)
    columns = list(set(list((dff.columns.values))) - set(l))

    for i in columns: 
        dff[i].fillna(newValue, inplace=True)

# A methoda to keep track of what columns that need to be converted to numerical values

def number_of_string_columns(df):
    columns = df.select_dtypes(include=["object"])
    number = 0
    string_columns = []
    for i in columns.columns:
        string_columns.append(i)
        number += 1
    
    return string_columns