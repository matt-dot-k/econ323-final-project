# Import packages and methods
import pandas as pd
from pandas import DataFrame as df
import numpy as np
import statsmodels as sm
from glmnet_py import glmnet
import xgboost as xgb

from plotnine import *

# Import and transform data
prop_prices_can = pd.read_csv("./data/can_property_prices.csv")
income_can = pd.read_csv("./data/can_income.csv")

prop_prices_can = (
    prop_prices_can.
    rename(columns = {"DATE": "Date", "QCAR628BIS": "Percent"})
)

prop_prices_can

income_can = (
    income_can[
        (income_can['Statistics'] == 'Median income (excluding zeros)')].
    filter(items = ['REF_DATE', 'VALUE']).
    rename(columns = {"REF_DATE": "Date", "VALUE": "Income"}).
    reset_index().
    drop('index', axis = 1)
)

base = income_can.loc[income_can['Date'] == 2010]['Income'].values[0]
income_can['Rebased'] = income_can['Income'] / base * 100

income_can

