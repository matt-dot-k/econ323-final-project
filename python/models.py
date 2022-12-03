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

# Preliminary plotting

plot_theme = theme(
    plot_title = element_text(
        face = "bold", size = 12, hjust = 0),
    plot_caption = element_text(
        size = 10, hjust = 0),
    axis_title_x = element_text(
        face = "bold", size = 12),
    axis_title_y = element_text(
        face = "bold", size = 12, angle = 90),
    axis_text = element_text(
        size = 12),
    panel_grid_major_x = element_blank(),
    panel_grid_minor_x = element_blank(),
    panel_grid_major_y = element_line(
        size = 0.1, color = "#808080"),
    panel_grid_minor_y = element_line(
        size = 0.1, color = "#808080"),
    panel_background = element_rect(
        fill = "#FFFFFF"),
    plot_background = element_rect(
        fill = "#FFFFFF"),
    legend_title = element_text(
        face = "bold", size = 12),
    legend_text = element_text(
        size = 12),
    legend_background = element_rect(
        fill = "#FFFFFF"),
    legend_key = element_rect(
        fill = "#FFFFFF")
)

price_chart = (ggplot(
    data = prop_prices_can,
    mapping = aes(
        x = 'Date', y = 'Percent', group = 1)
    ) +
    geom_line(
        color = '#F60552', size = 1.2
    ) +
    labs(
        x = 'Date', 
        y = 'Percentage (2010 = 100)',
        title = 'Residential Property Prices in Canada Since 2000'
    )
)

price_chart.draw()

income_chart = (ggplot(
    data = income_can) +
    geom_line(
        aes(x = 'Date', y = 'Rebased'),
        color = '#96CC28', size = 1.2
    ) +
    labs(
        x = 'Date',
        y = 'Percentage (2010 = 100)',
        title = 'Median Canadian Income Since 2000'
    )
)

income_chart.draw()

