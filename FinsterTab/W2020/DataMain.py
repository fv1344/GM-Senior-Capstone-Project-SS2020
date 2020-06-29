# Load other local Python modules to be used in this MAIN module
from FinsterTab.W2020.DataFetch import DataFetch
from FinsterTab.W2020.DataForecast import DataForecast
from FinsterTab.W2020.dbEngine import DBEngine
from FinsterTab.W2020.BuySell import BuySell
from FinsterTab.W2020.EngineeredFeatures import EngineeredFeatures
from FinsterTab.W2020.TradingSimulator import TradingSimulator
import FinsterTab.W2020.AccuracyTest

""" 
    Variables
"""
# instrument symbol table
instrument_master = 'dbo_instrumentmaster'
# Change to true if date dimension table is empty
reset_date_dim = False


"""
    Operations
"""

# Create database connection
db_engine = DBEngine().mysql_engine()

# Get raw market data
master_data = DataFetch(db_engine, instrument_master)

# Get ticker symbols
ticker_symbols = master_data.get_datasources()

# Get data from Yahoo! Finance and store in InstrumentStatistics
master_data.get_data(ticker_symbols)

my = DataForecast(db_engine, instrument_master)
my.calculate_william_forecast()
"""
# Get date data and store in DateDim, replaced the SQL calendar code
if reset_date_dim:
    master_data.get_calendar()


# Calculate forecast with functions that use macroeconomic indicators
FinsterTab.W2020.AccuracyTest.get_past_data(db_engine)
DataFetch.macroFetch(db_engine)
DataForecast.MSF1(db_engine)
DataForecast.MSF2(db_engine)
DataForecast.MSF3(db_engine)
DataForecast.MSF2_Past_Date(db_engine)

# Get raw data from database to calculate forecasts
indicators = EngineeredFeatures(db_engine, instrument_master)

# Calculate technical indicators and store in EngineeredFeatures
indicators.calculate()


# Get raw data from database to calculate forecasts
forecast = DataForecast(db_engine, instrument_master)


# Polynomial regression function
# Takes a while to run, comment out if need be
forecast.calculate_regression()


# calculate and store price predictions
forecast.calculate_forecast()


# calculate and store ARIMA forecast
forecast.calculate_arima_forecast()


# calculate and store Random Forest forecast
forecast.calculate_random_forest_forecast()


# flawed price prediction from previous semesters, without our improvements
forecast.calculate_forecast_old()


# calculate and store SVM forecast
forecast.calculate_svm_forecast()


# calculate and store XGBoost forecast
forecast.calculate_xgboost_forecast()


# Get Raw Data and Technical Indicators
signals = BuySell(db_engine, instrument_master)


# generate CMA trade signals
signals.cma_signal()


# generate FRL trade signals
signals.frl_signal()


# generate EMA trade signals
signals.ema_signal()


# generate MACD signals
signals.macd_signal()


# forecast-based signals
signals.algo_signal()


# Run Trade Simulations Based on Trade Signals
simulator = TradingSimulator(db_engine, instrument_master)


# individual strategies
simulator.trade_sim()


# combination trading strategy
simulator.combination_trade_sim()


# buy and hold simulation
simulator.buy_hold_sim()

"""
