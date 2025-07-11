import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco
import pandas_datareader as pdr
import datetime as dt
import talib as ta
import plotly.express as px
import time
import copy
import yfinance as yf


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

def getData(tickers, max_retries=5):
    start = dt.datetime(2016, 1, 1)
    end = dt.datetime(2025, 1, 1)

    for attempt in range(max_retries):
        try:
            data = yf.download(tickers, start=start, end=end, group_by='ticker', threads=False)
            if data.empty or 'Close' not in data:
                raise ValueError("Data is empty or missing 'Close' column")
            break  # Success
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(3 * (attempt + 1))  # Exponential backoff
    else:
        raise RuntimeError("Max retries reached. Failed to fetch data.")

    print("Data columns:", data.columns)
    data = data['Close']

    # Calculate technical indicators
    exp1 = data.ewm(span=12, adjust=False).mean()
    exp2 = data.ewm(span=26, adjust=False).mean()
    MACD = exp1 - exp2
    signal_line = MACD.ewm(span=9, adjust=False).mean()

    fifty = data.rolling(window=50).mean()
    twohundred = data.rolling(window=200).mean()

    return data, fifty, twohundred, MACD, signal_line

def getData(tickers):
    start = dt.datetime(2016, 1, 1)
    end = dt.datetime(2025, 1, 1)

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
    end = data.index.searchsorted(dt.datetime(curr_end_year, curr_end_month, 1))
    #new_data = data.iloc[start, end]
    new_data = data.iloc[start-1:end]#finds data for the time range provided
    #print(new_data.iloc[-1])
    #print(percentages)
    #print(initial_investment)
    market_value = np.sum((new_data/new_data.iloc[0])*percentages*initial_investment, axis=1)
    print("Return after", curr_year, "/", curr_month, ":", market_value[-1])
    return market_value[-1]

def accounting_for_movavg(percentages, fifty, twohundred, curr_year, curr_end_year, curr_month, curr_end_month):
    """
    Description: Calculates percentages accounting for the fifty and two hundred MAs
    Output: New normalized percentage list
    """
    try:
        fiftystart = fifty.index.searchsorted(dt.datetime(curr_year, curr_month, 1))
        twohundredstart = twohundred.index.searchsorted(dt.datetime(curr_year, curr_month, 1))

        if fiftystart >= len(fifty) or twohundredstart >= len(twohundred):
            raise IndexError("Date index out of bounds for moving averages. Possibly too recent.")

        new_fifty = fifty.iloc[fiftystart]
        new_twohundred = twohundred.iloc[twohundredstart]
    except IndexError as e:
        print(f"⚠️ Not enough data for moving average adjustments in {curr_year}-{curr_month}. Skipping MA adjustment.")
        return percentages  # Fall back to unadjusted percentages

    flag = (new_fifty - new_twohundred)/new_twohundred
    percentages_copy = copy.deepcopy(percentages)

    for i, val in enumerate(flag):
        if val < -0.01:
            percentages_copy[i] *= 0.5
        elif val > 0.01:
            percentages_copy[i] *= 1.5

    norm = [round(float(i)/sum(percentages_copy), 4) for i in percentages_copy]
    return norm

def macd_signal(MACD, signal_line, curr_year, curr_month, percentages):
    try:
        macd_val = MACD.index.searchsorted(dt.datetime(curr_year, curr_month, 1))
        signal_val = signal_line.index.searchsorted(dt.datetime(curr_year, curr_month, 1))

        if macd_val >= len(MACD) or signal_val >= len(signal_line):
            raise IndexError("Date index out of bounds for MACD or signal line.")

        new_macd = MACD.iloc[macd_val]
        new_signal = signal_line.iloc[signal_val]
    except IndexError as e:
        print(f"Not enough data for MACD signal adjustments in {curr_year}-{curr_month}. Skipping MACD adjustment.")
        return percentages  # Fall back to previous allocation

    percentages_copy = copy.deepcopy(percentages)
    for i in range(len(percentages)):
        if (new_macd[i] > new_signal[i]):
            percentages_copy[i] = percentages[i]*0.9
        elif (new_macd[i] < new_signal[i]):
            percentages_copy[i] = percentages[i]*1.1
    for i in range(len(percentages)):
        if (new_macd[i] > 0 and new_signal[i] > 0):
            percentages_copy[i] = percentages[i]*1.5

    norm = [round(float(i)/sum(percentages_copy), 4) for i in percentages_copy]
    return norm


def calculations(tickers, curr_year, initial_investment):
    data, fifty, twohundred, MACD, signal_line = getData(tickers)
    num_portfolios = 100000
    risk_free_rate = 0.0178
    count = 0
    curr_end_year = curr_year
    past_start = curr_year - 4
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
    static_y = []
    static_investment = initial_investment
    static_movavg = initial_investment
    dynamic_movavg = initial_investment
    static_macd_signal_investment = initial_investment
    dynamic_macd_signal_investment = initial_investment
    macd_signal_investment = initial_investment
    static_y.append(static_investment)
    static_ma = []
    dynamic_ma = []
    static_macd_signal = []
    dynamic_macd_signal = []
    normal_macd_signal = []
    static_ma.append(static_movavg)
    dynamic_ma.append(dynamic_movavg)
    static_macd_signal = [initial_investment]
    dynamic_macd_signal = [initial_investment]
    normal_macd_signal = [initial_investment]
    stock_num = len(tickers)

    #while count != 54:
    while not (curr_year==2025 and curr_month==1) :
        log_returns = setup(past_start, past_end, month, tickers, data)
        #print(log_returns)
        cov_matrix = log_returns.cov()
        mean_returns = log_returns.mean()
        percentages = display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate, log_returns, stock_num)
        movavg_percentages = accounting_for_movavg(percentages, fifty, twohundred, curr_year,curr_end_year, curr_month, curr_end_month)
        macd_signal_normal = macd_signal(MACD, signal_line, curr_year, curr_month, percentages)
        macd_signal_dynamic = macd_signal(MACD, signal_line, curr_year, curr_month, movavg_percentages)
        print(movavg_percentages, "ihihihihih")
        if count == 0:
            static_percentages = percentages
            #static_percentages = [1,0] #percentages #[1.0, 0, 0, 0, 0, 0, 0, 0, 0]  #
            #print("Static:",static_percentages)
        static_movavg_percent = accounting_for_movavg(static_percentages, fifty, twohundred, curr_year,curr_end_year, curr_month, curr_end_month)
        macd_signal_static = macd_signal(MACD, signal_line, curr_year, curr_month, static_percentages)
        print("Dynamic: ", end="")
        initial_investment = returns_calculation(percentages, initial_investment, curr_year, curr_end_year, curr_month, curr_end_month, data)
        print("Static: ", end="")
        static_investment = returns_calculation(static_percentages, static_investment, curr_year, curr_end_year, curr_month, curr_end_month, data)
        print("Static with moving average: ", end="")
        static_movavg = returns_calculation(static_movavg_percent, static_movavg, curr_year, curr_end_year, curr_month, curr_end_month, data)
        print("Dynamic with moving average: ", end="")
        dynamic_movavg = returns_calculation(movavg_percentages, dynamic_movavg, curr_year,curr_end_year, curr_month, curr_end_month, data)
        print("Static with macd_signal: ", end="")
        static_macd_signal_investment = returns_calculation(macd_signal_static, static_macd_signal_investment, curr_year, curr_end_year, curr_month, curr_end_month, data)
        print("macd_signal: ", end="")
        macd_signal_investment = returns_calculation(macd_signal_normal, macd_signal_investment, curr_year, curr_end_year, curr_month, curr_end_month, data)
        print("dynamic with macd_signal and movavg: ", end="")
        dynamic_macd_signal_investment = returns_calculation(macd_signal_dynamic, dynamic_macd_signal_investment, curr_year, curr_end_year, curr_month, curr_end_month, data)
        market_value.append(initial_investment)
        static_y.append(static_investment)
        static_ma.append(static_movavg)
        dynamic_ma.append(dynamic_movavg)
        static_macd_signal.append(static_macd_signal_investment)
        dynamic_macd_signal.append(dynamic_macd_signal_investment)
        normal_macd_signal.append(macd_signal_investment)
        month += 1
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
    return x_values, market_value, static_y, static_ma, dynamic_ma, static_macd_signal, dynamic_macd_signal, normal_macd_signal

def get_current_month_allocations(tickers, curr_year, curr_month):
    print(f"\nFetching data and calculating allocation for {curr_year}-{str(curr_month).zfill(2)}...\n")
    data, fifty, twohundred, MACD, signal_line = getData(tickers)
    past_start = curr_year - 4
    past_end = curr_year
    num_portfolios = 50000
    risk_free_rate = 0.0178
    stock_num = len(tickers)

    log_returns = setup(past_start, past_end, 1, tickers, data)
    cov_matrix = log_returns.cov()
    mean_returns = log_returns.mean()
    base_alloc = display_simulated_ef_with_random(
        mean_returns, cov_matrix, num_portfolios, risk_free_rate, log_returns, stock_num
    )

    ma_alloc = accounting_for_movavg(base_alloc, fifty, twohundred, curr_year, curr_year, curr_month, curr_month + 1)
    final_alloc = macd_signal(MACD, signal_line, curr_year, curr_month, ma_alloc)
    allocation_dict = {ticker: pct for ticker, pct in zip(tickers, final_alloc)}
    print("Recommended Portfolio Allocation:\n")
    for ticker, pct in allocation_dict.items():
        print(f" - {ticker}: {pct*100:.2f}%")

    return allocation_dict


if __name__ == "__main__":
    # Example usage
    # tickers = ['SPY', 'QQQ', 'IWM', 'VNQ']

    # try:
    #     year = int(input("Enter the current year (e.g., 2025): "))
    #     month = int(input("Enter the current month (1-12): "))
    #     assert 1 <= month <= 12
    # except (ValueError, AssertionError):
    #     print("❌ Invalid input. Please enter a valid year and month (1-12).")
    #     exit()

    # _ = get_current_month_allocations(tickers, year, month)


    tickers = ['SPY', 'QQQ', 'IWM', 'VNQ']  # You can change this to your desired tickers
    year = 2017
    initial_investment = 10000

    x_values, y_values, static_y, static_ma, dynamic_ma, static_macd_signal, dynamic_macd_signal, normal_macd_signal = calculations(
        tickers, year, initial_investment
    )

    # Plot using matplotlib
    plt.figure(figsize=(14, 8))
    plt.plot(x_values, y_values, marker='o', label='Dynamic')
    plt.plot(x_values, static_y, marker='o', label='Static')
    plt.plot(x_values, static_ma, marker='o', label='Static & MA')
    plt.plot(x_values, dynamic_ma, marker='o', label='Dynamic & MA')
    plt.plot(x_values, static_macd_signal, marker='o', label='Static & MACD Signal')
    plt.plot(x_values, dynamic_macd_signal, marker='o', label='Dynamic & MACD & MA')
    plt.plot(x_values, normal_macd_signal, marker='o', label='Dynamic & MACD Signal')

    plt.title('Portfolio Strategy Comparison Over Time')
    plt.xlabel('Months')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
