from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, FloatField, SelectField, SelectField, RadioField, BooleanField, SubmitField
from wtforms.validators import DataRequired, NumberRange

from app.data_fitting import DataFitting




class DataForm(FlaskForm):

    data_fitting = DataFitting()

    string_columns = data_fitting.getCategoricalColumns()

    column_formatted = []

    for name, values in string_columns:
        values_formatted = []
        for i in values:
            values_formatted.append(f"{i}")
        column_formatted.append([name, values_formatted])

    int_columns = data_fitting.getIntColumns()
    
    """
    for i in int_columns:
        print(f"{i} = IntegerField('{i}')")
    """
    
    MSZoning = SelectField(label=column_formatted[0][0], choices=column_formatted[0][1])
    Alley = SelectField(label=column_formatted[1][0], choices=column_formatted[1][1])
    LotShape = SelectField(label=column_formatted[2][0], choices=column_formatted[2][1])
    LandContour = SelectField(label=column_formatted[3][0], choices=column_formatted[3][1])
    LotConfig = SelectField(label=column_formatted[4][0], choices=column_formatted[4][1])
    LandSlope = SelectField(label=column_formatted[5][0], choices=column_formatted[5][1])
    Neighborhood = SelectField(label=column_formatted[6][0], choices=column_formatted[6][1])
    Condition1 = SelectField(label=column_formatted[7][0], choices=column_formatted[7][1])
    Condition2 = SelectField(label=column_formatted[8][0], choices=column_formatted[8][1])
    BldgType = SelectField(label=column_formatted[9][0], choices=column_formatted[9][1])
    HouseStyle = SelectField(label=column_formatted[10][0], choices=column_formatted[10][1])
    RoofStyle = SelectField(label=column_formatted[11][0], choices=column_formatted[11][1])
    RoofMatl = SelectField(label=column_formatted[12][0], choices=column_formatted[12][1])
    Exterior1st = SelectField(label=column_formatted[13][0], choices=column_formatted[13][1])
    Exterior2nd = SelectField(label=column_formatted[14][0], choices=column_formatted[14][1])
    MasVnrType = SelectField(label=column_formatted[15][0], choices=column_formatted[15][1])
    ExterQual = SelectField(label=column_formatted[16][0], choices=column_formatted[16][1])
    ExterCond = SelectField(label=column_formatted[17][0], choices=column_formatted[17][1])
    Foundation = SelectField(label=column_formatted[18][0], choices=column_formatted[18][1])
    BsmtQual = SelectField(label=column_formatted[19][0], choices=column_formatted[19][1])
    BsmtCond = SelectField(label=column_formatted[20][0], choices=column_formatted[20][1])
    BsmtExposure = SelectField(label=column_formatted[21][0], choices=column_formatted[21][1])
    BsmtFinType1 = SelectField(label=column_formatted[22][0], choices=column_formatted[22][1])
    BsmtFinType2 = SelectField(label=column_formatted[23][0], choices=column_formatted[23][1])
    Heating = SelectField(label=column_formatted[24][0], choices=column_formatted[24][1])
    HeatingQC = SelectField(label=column_formatted[25][0], choices=column_formatted[25][1])
    CentralAir = SelectField(label=column_formatted[26][0], choices=column_formatted[26][1])
    Electrical = SelectField(label=column_formatted[27][0], choices=column_formatted[27][1])
    KitchenQual = SelectField(label=column_formatted[28][0], choices=column_formatted[28][1])
    Functional = SelectField(label=column_formatted[29][0], choices=column_formatted[29][1])
    FireplaceQu = SelectField(label=column_formatted[30][0], choices=column_formatted[30][1])
    GarageType = SelectField(label=column_formatted[31][0], choices=column_formatted[31][1])
    GarageFinish = SelectField(label=column_formatted[32][0], choices=column_formatted[32][1])
    GarageQual = SelectField(label=column_formatted[33][0], choices=column_formatted[33][1])
    GarageCond = SelectField(label=column_formatted[34][0], choices=column_formatted[34][1])
    PavedDrive = SelectField(label=column_formatted[35][0], choices=column_formatted[35][1])
    PoolQC = SelectField(label=column_formatted[36][0], choices=column_formatted[36][1])
    Fence = SelectField(label=column_formatted[37][0], choices=column_formatted[37][1])
    SaleType = SelectField(label=column_formatted[38][0], choices=column_formatted[38][1])
    SaleCondition = SelectField(label=column_formatted[39][0], choices=column_formatted[39][1])
    
    
    MSSubClass = IntegerField('MSSubClass')
    LotFrontage = IntegerField('LotFrontage')
    LotArea = IntegerField('LotArea')
    OverallQual = IntegerField('OverallQual')
    OverallCond = IntegerField('OverallCond')
    YearBuilt = IntegerField('YearBuilt')
    YearRemodAdd = IntegerField('YearRemodAdd')
    MasVnrArea = IntegerField('MasVnrArea')
    BsmtFinSF1 = IntegerField('BsmtFinSF1')
    BsmtFinSF2 = IntegerField('BsmtFinSF2')
    BsmtUnfSF = IntegerField('BsmtUnfSF')
    TotalBsmtSF = IntegerField('TotalBsmtSF')
    _1stFlrSF = IntegerField('1stFlrSF')
    _2ndFlrSF = IntegerField('2ndFlrSF')
    LowQualFinSF = IntegerField('LowQualFinSF')
    GrLivArea = IntegerField('GrLivArea')
    BsmtFullBath = IntegerField('BsmtFullBath')
    BsmtHalfBath = IntegerField('BsmtHalfBath')
    FullBath = IntegerField('FullBath')
    HalfBath = IntegerField('HalfBath')
    BedroomAbvGr = IntegerField('BedroomAbvGr')
    KitchenAbvGr = IntegerField('KitchenAbvGr')
    TotRmsAbvGrd = IntegerField('TotRmsAbvGrd')
    Fireplaces = IntegerField('Fireplaces')
    GarageYrBlt = IntegerField('GarageYrBlt')
    GarageCars = IntegerField('GarageCars')
    GarageArea = IntegerField('GarageArea')
    WoodDeckSF = IntegerField('WoodDeckSF')
    OpenPorchSF = IntegerField('OpenPorchSF')
    EnclosedPorch = IntegerField('EnclosedPorch')
    _3SsnPorch = IntegerField('3SsnPorch')
    ScreenPorch = IntegerField('ScreenPorch')
    PoolArea = IntegerField('PoolArea')
    MoSold = IntegerField('MoSold')
    YrSold = IntegerField('YrSold')
    
  

    submit = SubmitField('Submit')