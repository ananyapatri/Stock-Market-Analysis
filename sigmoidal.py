import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco
import pandas_datareader as pdr
import datetime as dt
import dash
from dash import dcc
from dash import html
import talib as ta
from talib import RSI, BBANDS
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import time
import copy

def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]

def efficient_return(mean_returns, cov_matrix, target):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    def portfolio_return(weights):
        return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[1]

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficients = []
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, cov_matrix, ret))
    return efficients

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

import yfinance as yf
import datetime as dt

import yfinance as yf
import datetime as dt

def getData(tickers):
    start = dt.datetime(2016, 1, 1)
    end = dt.datetime(2022, 8, 1)

    try:
        # Download the data
        data = yf.download(tickers, start=start, end=end)
    except Exception as e:
        print(f"Error fetching data: {e}")
        data = None

    if data is not None:
        print("Data columns:", data.columns)  # Print the column names to check for 'Close'

        # Extract the 'Close' prices for each ticker (using MultiIndex)
        data = data['Close']

        # Calculate exponential moving averages (EMA)
        exp1 = data.ewm(span=12, adjust=False).mean()
        exp2 = data.ewm(span=26, adjust=False).mean()
        MACD = exp1 - exp2
        signal_line = MACD.ewm(span=9, adjust=False).mean()

        # Calculate moving averages (50-day and 200-day)
        fifty = data.rolling(window=50).mean()
        twohundred = data.rolling(window=200).mean()

        return data, fifty, twohundred, MACD, signal_line
    else:
        return None, None, None, None, None



def setup(startyear, endyear, month, tickers, data):
    """
    Description: For the time range provided, calculated a dataframe of log Returns
    Output: log_returns
    """
    start = data.index.searchsorted(dt.datetime(startyear, month, 1))
    end = data.index.searchsorted(dt.datetime(endyear, month, 1))
    data = data.iloc[start:end]#finds data for the time range provided
    log_returns = np.log(data/data.shift())
    return log_returns

def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) *252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns

def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate, log_returns, stock_num):
    results = np.zeros((3,num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(stock_num)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record

def display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate, log_returns, stock_num):
    """
    Description: prints the Maximum Sharpe Ratio Portfolio Allocation, annualised return, and annualised volatility.
    Input: mean_returns, cov_matrix, num_portfolios, risk_free_rate, log_returns
    Output: A list of percentages of how much to invest in each stock
    """
    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = portfolio_annualised_performance(max_sharpe['x'], mean_returns, cov_matrix)
    target_return=efficient_return(mean_returns, cov_matrix, 1.5*rp)


    max_sharpe_allocation = pd.DataFrame(max_sharpe.x,index=log_returns.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]

    money = []
    for i in max_sharpe_allocation.allocation:
        amount = round((i/100.0),4)#adds each percentage to the list, rounds to 4 decimal places
        money.append(amount)

    max_sharpe_allocation = max_sharpe_allocation.T

    """
    target_return_allocation = pd.DataFrame(target_return.x,index=log_returns.columns,columns=['allocation'])
    target_return_allocation.allocation = [round(i*100,2)for i in target_return_allocation.allocation]

    money = []
    for i in target_return_allocation.allocation:
        amount = round((i/100.0),4)#adds each percentage to the list, rounds to 4 decimal places
        money.append(amount)

    target_return_allocation = target_return_allocation.T
    """
    return money

def returns_calculation(percentages, initial_investment, curr_year, curr_end_year, curr_month, curr_end_month, data):
    """
    Description: Invests initial_investment in 4 stocks based on a percentage
    calculates the change in what was invested after each day
    output: Change in the investment after each month
    """
    start = data.index.searchsorted(dt.datetime(curr_year, curr_month, 1))
    start = start - 1
    end = data.index.searchsorted(dt.datetime(curr_end_year, curr_end_month, 1))
    new_data = data.iloc[start:end]#finds data for the time range provided
    market_value = np.sum((new_data/new_data.iloc[0])*percentages*initial_investment, axis=1)
    print("Return after", curr_year, "/", curr_month, ":", market_value[-1])
    return market_value[-1]

def accounting_for_movavg(percentages, fifty, twohundred, curr_year,curr_end_year, curr_month, curr_end_month):
    """
    Description: Calculates percentages accounting for the fifty and two hundred MAs
    Output: New normalized percentage list
    """
    fiftystart = fifty.index.searchsorted(dt.datetime(curr_year, curr_month, 1))#gets the data row number for fifty day moving average
    twohundredstart = twohundred.index.searchsorted(dt.datetime(curr_year, curr_month, 1))
    new_fifty = fifty.iloc[fiftystart]#accesses row for fifty day moving average
    new_twohundred = twohundred.iloc[twohundredstart]
    flag = (new_fifty - new_twohundred)/new_twohundred#checks to see if the moving averages differ by a distinct margin
    counter = 0
    values = [0]*len(percentages)
    percentages_copy = copy.deepcopy(percentages)
    for i in range(len(flag)):
        if flag[i] < -0.01:
            values[i] -= 1
        elif flag[i] > 0.01:
            values[i] += 1
    return values

def macd_signal(MACD, signal_line, curr_year, curr_month, percentages):
    macd_val = MACD.index.searchsorted(dt.datetime(curr_year, curr_month, 1))
    signal_val = signal_line.index.searchsorted(dt.datetime(curr_year, curr_month, 1))
    new_macd = MACD.iloc[macd_val]
    new_signal = signal_line.iloc[signal_val]
    values = [0]*len(percentages)#[0,0,0,0]
    for i in range(len(percentages)):
        if (new_macd[i] > new_signal[i]):
            values[i] -= 1
        elif (new_macd[i] < new_signal[i]):
            values[i] += 1
    for i in range(len(percentages)):
        if (new_macd[i] > 0 and new_signal[i] > 0):
            values[i] += 1
    return values

def sigmoidal_function(sigmoidal_values, percentages):
    """
    Description: uses the function- L/(1 + e^(-k(x - xo)))
    where L is the curve's maximum value
    xo is the x-value of the sigmoid's midpoint
    k is the logistic growth rate(Steepness) of the curve
    Output: new normalized percentage list
    """
    percentages_copy = copy.deepcopy(percentages)
    for i in range(len(sigmoidal_values)):
        weightage = 3/(1 + 2.71828**(-1*sigmoidal_values[i]))
        percentages_copy[i] = percentages[i]*weightage
    norm = [round(float(i)/sum(percentages_copy), 4) for i in percentages_copy]#normalizes the list so they add to 1
    return norm

def bollingerbands(new_data, curr_year, curr_month, percentages, row):
    values = [0]*len(percentages)
    for i in range(len(percentages)):
        up, mid, low = BBANDS(new_data[ :,i], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        if(new_data[row,i] > up[row]):
            values[i] += 1
        if(new_data[row,i] < low[row]):
            values[i] -= 1
    print(values)
    return values

def calculations(tickers, curr_year, initial_investment):
    data, fifty, twohundred, MACD, signal_line = getData(tickers)
    new_data = data.to_numpy()
    print(new_data)
    sigmoidal_values = []
    num_portfolios = 25000
    risk_free_rate = 0.0178
    count = 0
    curr_end_year = curr_year
    past_start = curr_year - 2
    past_end = curr_year
    month = 1
    counter = 1
    curr_month = 1
    curr_end_month = 2
    market_value = []
    market_value.append(initial_investment)
    x_values = []
    x_values.append(counter)
    log_returns = setup(past_start, past_end, month, tickers, data)
    stock_num = len(tickers)


    while not(curr_year == 2022 and curr_month == 8):
        row = data.index.searchsorted(dt.datetime(curr_year, curr_month, 1))
        log_returns = setup(past_start, past_end, month, tickers, data)
        cov_matrix = log_returns.cov()
        mean_returns = log_returns.mean()
        percentages = display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate, log_returns, stock_num)
        if count == 0:
            static_percentages = percentages
        #sigmoidal data under this
        value1 = macd_signal(MACD, signal_line, curr_year, curr_month, percentages)#[0, -1, 1, 2]
        value2 = accounting_for_movavg(percentages, fifty, twohundred, curr_year,curr_end_year, curr_month, curr_end_month)#[-1, 2, 1, 0]
        value3 = bollingerbands(new_data, curr_year, curr_month, percentages, row)
        sigmoidal_values = [0]*len(tickers)
        for i in range(len(sigmoidal_values)):
            sigmoidal_values[i] = sigmoidal_values[i] + value1[i] + value2[i] + value3[i]
        sigmoidal_percentages = sigmoidal_function(sigmoidal_values, percentages)
        initial_investment = returns_calculation(sigmoidal_percentages, initial_investment, curr_year, curr_end_year, curr_month, curr_end_month, data)
        market_value.append(initial_investment)
        month += 1
        if month == 13:
            month = 1
            past_start += 1
            past_end += 1
        curr_month += 1
        curr_end_month += 1
        if curr_end_month == 13:
            curr_end_month = 1
            curr_end_year += 1
        if curr_month == 13:
            curr_month = 1
            curr_year += 1
        count += 1
        counter += 1
        x_values.append(counter)
    return x_values, market_value

if __name__ == "__main__" :
    app = Dash(__name__)

    app.layout = html.Div([
	html.H6("Investment Outlook"),
        dcc.Graph(id='graph-with-slider'),
	html.Br(),
        html.Div([
        " stocks: ",
        dcc.Input( type='text', value="AAPL NFLX GOOGL AMZN", id='stock-slider'),
        " Current year: ",
        dcc.Input( type='number', value=2019, id='current-year-slider'),
        " Initial investment: ",
        dcc.Input( type='number', value=100000, id='investment-slider'),
        html.Button('Submit', id='submit-val', n_clicks=0)
	]),
    ])
    @app.callback(
	Output('graph-with-slider', 'figure'),
	[State('stock-slider', 'value'),
    State('current-year-slider', 'value'),
	State('investment-slider', 'value')],
    Input('submit-val', 'n_clicks'))

    def update_figure( stocks, year, investment, n_clicks):
        if n_clicks is None:
            raise PreventUpdate
        else:
            stocks = stocks.split(' ')
            print(stocks)
            x_values, y_values = calculations(stocks, year, investment)
            fig = px.line(x=x_values, y=y_values, template="plotly_dark", markers = True,
            labels={'x':'months', 'y':'return', }) # override keyword names with labels
            #newnames = {'wide_variable_0':'Dynamic'}
            #fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
            #                                      legendgroup = newnames[t.name],
            #                                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])
            #                                     )
            #                  )

            fig.update_layout(transition_duration=500)
            return fig

    if __name__ == "__main__":
        app.run_server(debug=True)
