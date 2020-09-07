# Basic Functions:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import requests
import time
import warnings
import datetime

# SQL
from sqlalchemy import create_engine

# Special Functions:
from scipy import optimize
from dateutil.relativedelta import relativedelta
from scipy.stats import linregress as LR
from scipy.stats import norm
from scipy.stats import gennorm
from scipy.stats import kstest
from scipy.stats import t

# SQL CONNECTION
eng_risk = create_engine("mysql+pymysql://root:Rafa1574@localhost/risk")
eng_fund = create_engine("mysql+pymysql://root:Rafa1574@localhost/fundamentalista")
eng_info = create_engine("mysql+pymysql://root:Rafa1574@localhost/info")
con = eng_risk.connect()
con_fund = eng_fund.connect()
con_info = eng_info.connect()

# Reference Dates:
today = datetime.datetime.today()
today = today.replace(hour = 0,minute = 0, second=0, microsecond = 0)
year_1 = today + relativedelta(years = -1)
days_30 = today + relativedelta(days = -30)
days_60 = today + relativedelta(days = -60)


## LISTA FUNÇÕES 2.0:
############ QUOTE ################
# YAHOO_QUOTE
# SQL_QUOTE
# DAILY_UPDATE

############ SPECIAL ##############
# DIST_TEST
# COVARIANCE

############ STOCK ################
# UPDATE
# BETA
# SELF_CORR
# IND_RISK (ADD 'end = today' for backtesting)


############ PORTFOLIO ############
# COV_MATRIX
# CHOLESKY
# RISK_P (MARKOWITZ)
# OPTIMIZATION (MARKOWITZ) | OTHER METHODS MISSING
# HIST_RETURNS

############# VaR ################



############# QUOTE ##############
# get quote from Yahoo Finance
def yahoo_quote(symbol='BVSP', start='2010-01-01', end=today):
    if symbol == 'BVSP':
        symbol = '^' + symbol
    elif symbol == 'bvsp':
        symbol = '^' + symbol.upper()
    else:
        symbol = symbol + '.SA'

    stock = pdr.DataReader(symbol, start=start, end=end, data_source='yahoo')
    return stock

# Get quote from Local DB
def sql_quote(ticker = 'bvsp'):
    query = 'select * from ' + ticker
    df = pd.read_sql(query, con = con)
    return df

# Update all quotes in DB
def daily_update():
    hour = datetime.datetime.today().hour
    if hour <= 18:
        print('Market may not be closed yet')
        return None
    else:
        stocks = pd.read_sql("select * from beta", con = con_info)
        stocks = stocks['Ticker'].tolist()
        for st in stocks:
            stock = Stock(st)
            stock.update()
            time.sleep(5)

############# SPECIAL ############

# Estudo de Correlação:
def covariance(ticker1, ticker2):
        # Load and pair Stock data from DB
        stock1 = Stock(ticker1)
        stock2 = Stock(ticker2)
        df1 = stock1.data
        df2 = stock2.data
        df1['pair'] = df2['return']
        df1.dropna(inplace = True)
        df1 = df1[df1.index >= year_1]
        # return covariance of pair
        return np.cov(df1['return'], df1['pair'])[0][1]

# Optimization of Risk given Return. Methods: Markowitz or [missing] Portfolio Agregation(STD(30,60), EWMA, GED)
def optimization(port, method = 'mark'):
    stocks = port.stocks
    cols = ['return', 'risk']
    for st in stocks:
        cols.append(st)
    # Create Summary table for risk analysis
    result = pd.DataFrame(columns = cols, index = range(99))
    # Define interval of possible returns (No Short Sell)
    r = np.array(port.ind_return)
    r_max = max(port.ind_return)
    r_min = min(port.ind_return)
    step = (r_max - r_min)/100
    for i in range(99):
        result['return'][i] = r_min + i*step
    # Optimization Set UP: Return Constraint; Bounds for weights; Initial Guess (W_0):
    # Initial Guess:
    w_0 = np.ones(len(stocks))/len(stocks)
    
    # Bounds for weights (No Short Sell):
    bounds = []
    for st in stocks:
        bounds.append((0.0,1.0))
    bounds = tuple(bounds)
    
    # Constraint1: Sum of weights = 1
    def constraint1(weights):
        return np.sum(weights) - 1
    
    # Return constraint defined on loop:
    for i in range(99):
        r_int = result['return'][i]
        def constraint2(weights):
            return np.dot(weights, r) - r_int
        
    # Optimization:
        constraints = [{'type':'eq', 'fun': constraint1}, {'type':'eq', 'fun': constraint2}]
        solver = optimize.minimize(port.risk_p, w_0, constraints = constraints, bounds = bounds)
        optimal_w = solver['x']
        for j in range(len(stocks)):
            result[stocks[j]][i] = optimal_w[j]
        result['risk'][i] = port.risk_p(optimal_w)
    return result


# Test Distribution: log = Series of Returns (usually 1 year of history)
# Methos = Normal Distribution "norm"; Generalized Error Distribution "ged";
def dist_test(log, conf_level = 0.95, dist = 'normal'):
    # test for normal
    if dist == 'normal':
        rv = norm(loc = np.mean(log), scale = np.std(log, ddof = 1))
        test = kstest(log, rv.cdf)
        D_crit = 1.3581/np.sqrt(len(log))
        # test result
        if test[0] > D_crit or test[1] < 1 - conf_level:
            result = False
        else:
            result = True
        return result
    # test for ged
    elif dist == 'ged':
        beta = 1.3
        log2 = [abs(x)**beta for x in log]
        scale = (np.mean(log2)*beta)**(1/beta)
        rv = gennorm(beta = beta, scale = scale)
        D_crit = 1.3581/np.sqrt(len(log))
        #test result
        if test[0] > D_crit or test[1] < 1 - conf_level:
            result = False
        else:
            result = True
        return result
    else:
        return None

# 
class Stock:
    def __init__(self, ticker):
        # Getting Data from SQL:
        try:
            df = sql_quote(ticker)
            df['return'] = np.log(df['Adj Close']/df['Adj Close'].shift(1))
            df.set_index('Date', inplace = True)
            
        # If not possible get data from Yahoo_Finance:
        except Exception:
            df = yahoo_quote(symbol = ticker)
            df.reset_index(inplace = True)
            # Save non_existing data to SQL:
            df.to_sql(name = ticker, con = con, if_exists = 'replace', index = False)
        
            df['return'] = np.log(df['Adj Close']/df['Adj Close'].shift(1))
            df.set_index('Date', inplace = True)
        
        # Load Basic Info:
        self.ticker = ticker
        self.data = df
        self.start_date = df.index[0]
        self.end_date = df.index[-1]
        self.Beta = {}
        self.corr_ret = {}
    
    def update(self):
        df = self.data
        hour = datetime.datetime.today().hour
        if hour <= 18:
            end = today + relativedelta(days = -1)
        else: end = today
        start = end + relativedelta(days = -5)
        df_new = yahoo_quote(symbol = self.ticker, start = start, end = end)
        for d in df_new.index:
            if d in list(df.index):
                df_new['High'][d] = np.nan
        df_new.dropna(inplace = True)
        df_new.reset_index(inplace = True)
        df_new.to_sql(name = self.ticker, con = con, if_exists = 'append', index = False)
        df = sql_quote(self.ticker)
        df['return'] = np.log(df['Adj Close']/df['Adj Close'].shift(1))
        self.data = df
        return None
    
    # BETA Study:
    def beta(self):
        bova = Stock('bvsp').data
        bova['stock'] = self.data['return']
        bova.dropna(inplace = True)
        bova = bova[bova.index >= year_1]
        result = LR(bova['return'], bova['stock'])
        self.Beta['beta'] = result[0]
        self.Beta['alpha'] = result[1]
        self.Beta['R^2'] = result[2]**2
        return self.Beta
    
    # Conditional Return:
    def self_corr(self):
        corr = self.data.copy()
        corr['past'] = corr['return'].shift(1)
        corr.dropna(inplace = True)
        corr = corr[corr.index>=year_1]
        result = LR(corr['past'], corr['return'])
        self.corr_ret['beta'] = result[0]
        self.corr_ret['alpha'] = result[1]
        self.corr_ret['R^2'] = result[2]**2
        return self.corr_ret

    # Individual Risk Evaluation:
    def ind_risk(self, method = 'norm', start = year_1, end = today, conf_level = 0.95, beta = 1.3, expo = 0.94):
        """
        Common methods include: 
            - "norm": Normal (Guassian) with 1 year period, 60 days periods, 30 days periods
            - "student": T of Student with 60 days periods, 30 days periods
            - Generalized Error with 1 year period, 60 days periods, 30 days periods
                Inform BETA
            - EWMA with lambda = 0.94 or lambda = 0.97
                Inform factor
            - Level of Confidence = 0.95 or 0.99
        """
        if method == 'norm':
            df = self.data
            df = df[df.index >= start]
            df = df[df.index <= end]
            R =  df['return'].mean()
            S = np.std(df['return'], ddof = 1)
            rv = norm(loc = R, scale = S)
            var = rv.ppf(1 - conf_level)
            return var

        elif method == 'student':
            df = self.data
            df = df[df.index >= start]
            df = df[df.index <= end]
            R = df['return'].mean()
            S = np.std(df['return'], ddof = 1)
            rv = t(loc = R, scale = S, df = df.shape[0] - 1)
            var = rv.ppf(1 - conf_level)
            return var

        elif method == 'ged':
            df = self.data
            df = df[df.index >= start]
            df = df[df.index <= end]
            log = df['return']
            log2 = [abs(x)**beta for x in log]
            scale = (np.mean(log2)*beta)**(1/beta)
            rv = gennorm(beta = beta, scale = scale)
            var = rv.ppf(1 - conf_level)
            return var

        elif method == 'ewma':
            df = self.data
            df = df[df.index <= end]
            log = df['return']
            log = pd.DataFrame(log)
            index = list(log.index)
            log['factor'] = [len(index) - index.index(d) - 1 for d in log.index]
            log['weight'] = (1-expo)*(expo**log['factor'])
            log['ewma_return'] = log['return']*log['weight']
            log['ewma_std'] = (log['return']**2)*log['weight']
            R = log['ewma_return'].sum()
            S = np.sqrt(log['ewma_std'].sum())
            rv = norm(loc = R, scale = S)
            var = rv.ppf(1 - conf_level)
            return var
        else:
            print('Undefined Method')
            return None


# Portfolio analysis:
class Portfolio:
    def __init__(self, stocks):
        self.stocks = stocks
    # Return and Individual Risk
        R = []
        S = []
        self.data = []
        for st in stocks:
            df = Stock(st).data
            self.data.append(df)
            R.append(df[df.index >= year_1]['return'].mean())
            S.append(df[df.index >= year_1]['return'].std())
        self.ind_return = R
        self.ind_risk = S

    # Covariance Matrix
        cov_matrix = pd.DataFrame(columns = stocks, index = stocks)
        # Iterate to calculate Cov_Matrix
        for st1 in cov_matrix.columns:
            for st2 in cov_matrix.index:
                cov_matrix[st1][st2] = covariance(st1, st2)
        self.cov_matrix = cov_matrix
        # Cholesky Decomposition for Cov_Matrix
        self.cholesky = np.linalg.cholesky(np.matrix(cov_matrix, dtype = float))

    # Historic Return for Portfolio    
    def hist_returns(self, weights, start = year_1):
        if len(weights) == len(self.stocks):
            cap = 100000
            Q = []
            for i in range(len(self.stocks)):
                Pi = self.data[i]['Adj Close'][self.data[i].index[-1]]
                Q.append(weights[i]*cap/Pi)
            index = [start + relativedelta(days = i) for i in range((today-year_1).days)]
            result = pd.DataFrame(columns = ['port_price', 'port_return'], index = index)
            for d in result.index:
                P = []
                for i in range(len(self.stocks)):
                    try:
                        P.append(Q[i]*self.data[i]['Adj Close'][d])
                    except Exception:
                        P.append(np.nan)
                result['port_price'][d] = np.sum(P)
            result['port_price'] = result['port_price'].astype(float)
            result.dropna(subset = ['port_price'], inplace = True)
            result['port_return'] = np.log(result['port_price']/result['port_price'].shift(1))
            result.dropna(inplace = True)
            return result
        else:
            return None
        

    # Portfolio Risk:
    def risk_p(self, weights):
        weights = np.array(weights)
        r = np.array(self.ind_return)
        if len(r) == len(weights):
            R = np.dot(weights, r)
            var = np.dot(weights, np.dot(self.cov_matrix, weights.transpose()))
            std_port = np.sqrt(var)
            return std_port
        else: return None