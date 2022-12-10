# Import packages for data handling and plotting
import pandas as pd
from pandas import DataFrame as df
import numpy as np
import matplotlib.pyplot as plt
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
    panel_border = element_rect(
        fill = "#000000", size = 1.0),
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

# Import packages for pre-processing data and model fitting
from sklearn import model_selection, linear_model
import statsmodels.api as sm
from skranger.ensemble import RangerForestRegressor

boston = pd.read_csv("./data/boston.csv")

X = boston.drop('MEDV', axis = 1)
y = boston['MEDV']

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size = 0.25, random_state = 50)

# Fit ordinary least squares model 
OLS_model = sm.OLS(endog = y_train, exog = X_train)
OLS_fit = OLS_model.fit()

# Fit L1 regularized regression
lasso_model = linear_model.Lasso(alpha = 1.0, warm_start = True)
lasso_fit = lasso_model.fit(X = X_train, y = y_train)

a = np.exp(np.linspace(10, -2, 50))
alphas, coefs_lasso, _ = linear_model.lasso_path(
    X_train, y_train, alphas = a, max_iter = 10000)

def lasso_path(X, y, alpha):
    """
    Takes in a matrix of regressors (X), a vector of response 
    values (y), and a vector of regularization coefficients (alphas)

    Obtains lasso coefficients for each iteration over alphas and
    plots the resulting coefficient trace in ggplot style
    """

    # Obtain coefficients across the whole lasso path
    coefs = []
    for a in alpha:
        lasso = linear_model.Lasso(alpha = a, fit_intercept = False)
        lasso.fit(X, y)
        coefs.append(lasso.coef_)
    
    # Create dataframe for plotting the coefficient path
    alpha_coefs = pd.DataFrame(coefs)
    alpha_coefs['alpha'] = alphas
    colors = [
        '#FF8833', '#00A0DD', '#593380', '#0F5499', '#00994D', '#CC1451', 
        '#CCE6FF', '#990F3D', '#0D7680', '#0F5499', '#593380', '#FF7FAA', "#0D7680"]

    # Plot the coefficient path 
    coef_plot = ggplot()
    for i in (range(len(list(alpha_coefs.columns)) - 1)):
        coef_plot = (
            coef_plot + 
            geom_line(
                data = alpha_coefs,
                mapping = aes(x = 'alpha', y = alpha_coefs[i]),
                size = 1.2,
                color = colors[i]) +
            scale_x_log10() +
            plot_theme)
    
    coef_plot = (
        coef_plot +
        labs(
            x = "Alpha",
            y = "Coefficients",
            title = "Path of Lasso Coefficients with Varying Alpha")) 
    
    return coef_plot

lasso_path(X = X_train, y = y_train, alpha = alphas)

# Comparing the coefficient estimates of OLS and lasso
OLS_coefs = OLS_fit.params
lasso_coefs = lasso_fit.coef_

coef_table = pd.DataFrame()
coef_table['OLS'] = OLS_coefs
coef_table['lasso'] = lasso_coefs