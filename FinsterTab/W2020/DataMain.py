# Load other local Python modules to be used in this MAIN module
from FinsterTab.W2020.DataFetch import DataFetch
from FinsterTab.W2020.DataForecast import DataForecast
from FinsterTab.W2020.dbEngine import DBEngine
from FinsterTab.W2020.BuySell import BuySell
from FinsterTab.W2020.EngineeredFeatures import EngineeredFeatures
from FinsterTab.W2020.TradingSimulator import TradingSimulator
import FinsterTab.W2020.AccuracyTest

# instrument symbol table
instrument_master = 'dbo_instrumentmaster'

"""
    TOGGLE THE FOLLOWING BOOLEANS TO RUN THE DESIRED PORTIONS OF THE APPLICATION
    Phase 1: Set up db
    Phase 2: Engineered data
    Phase 3: Predictive data
    Phase 4: Signals
    Phase 5: Simulation
"""
update_close_stats = False              # Pass 1.1
reset_date_dim = False                  # Pass 1.2  (Takes around 1 minute)
update_macro_stats = False              # Pass 1.3
update_msf_forecast = False             # Pass 3.2  (Takes around 3 minutes)
update_engineered_features = False      # Pass 2
update_remaining_forecasts = False      # Pass 3.1  (Takes around 1 hour. Saving "old forecasts" is paradoxical)
update_signals = False                  # Pass 4    (Takes around 5-10 minutes
run_simulator = False                   # Pass 5    (Takes around 15 minutes)
update_ars_forecast = False             # Pass 3.3

"""
    OPERATIONS BELOW
"""

# Create database connection
db_engine = DBEngine().mysql_engine()

if update_close_stats:
    # Get raw market data
    master_data = DataFetch(db_engine, instrument_master)
    # Get ticker symbols
    ticker_symbols = master_data.get_datasources()
    # Get data from Yahoo! Finance and store in InstrumentStatistics
    master_data.get_data(ticker_symbols)

# Get date data and store in DateDim, replaced the SQL calendar code
if reset_date_dim:
    reset_calender = DataFetch(db_engine, instrument_master)
    reset_calender.get_calendar("2000-01-01", "2025-12-31")

# Calculate forecast with functions that use macroeconomic indicators
if update_macro_stats:
    FinsterTab.W2020.AccuracyTest.get_past_data(db_engine)
    DataFetch.macroFetch(db_engine)

if update_msf_forecast:
    DataForecast.MSF1(db_engine)
    DataForecast.MSF2(db_engine)
    DataForecast.MSF3(db_engine)
    DataForecast.MSF2_Past_Date(db_engine)

if update_engineered_features:
    # Get raw data from database to calculate forecasts
    indicators = EngineeredFeatures(db_engine, instrument_master)
    # Calculate technical indicators and store in EngineeredFeatures
    indicators.calculate()

if update_remaining_forecasts:
    # Get raw data from database to calculate forecasts
    forecast = DataForecast(db_engine, instrument_master)

    # Polynomial regression function
    # Takes a while to run, comment out if need be
    # Params: start regression analysis date, number of days to use for analysis, forecast amount of days
    # Flawed implementation because every record is deleted and recomputed every run
    # 1 year = 1.5 min runtime
    # Each run will take the same amount of time
    forecast.calculate_regression("2019-07-15", 20, 5)

    # calculate and store price predictions ("PricePred")
    # 10 years = 2 min runtime
    # Coded to not overwrite whole table every time (i.e. each run after first will be quick)
    forecast.calculate_forecast()

    # flawed price prediction from previous semesters, without our improvements ("PricePredOld")
    # 10 years = 2 min runtime
    # Coded to not overwrite whole table every time (i.e. each run after first will be quick)
    forecast.calculate_forecast_old()

    # calculate and store ARIMA forecast
    # 10 years = 3 min runtime
    # Coded to not overwrite whole table every time (i.e. each run after first will be quick)
    forecast.calculate_arima_forecast()

    # calculate and store Random Forest forecast
    # 1 year = 3.5min runtime
    # Coded to not overwrite whole table every time (i.e. each run after first will be quick)
    forecast.calculate_random_forest_forecast("2019-07-15")

    # calculate and store SVM forecast
    # 10 years = 2.5 min runtime
    # Coded to not overwrite whole table every time (i.e. each run after first will be quick)
    forecast.calculate_svm_forecast()

    # calculate and store XGBoost forecast
    # 10 years = 3.5 min runtime
    # Coded to not overwrite whole table every time (i.e. each run after first will be quick)
    forecast.calculate_xgboost_forecast()

if update_signals:
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

if run_simulator:
    # Run Trade Simulations Based on Trade Signals
    simulator = TradingSimulator(db_engine, instrument_master)
    # individual strategies
    simulator.trade_sim()
    # combination trading strategy
    simulator.combination_trade_sim()
    # buy and hold simulation
    simulator.buy_hold_sim()

if update_ars_forecast:
    my = DataForecast(db_engine, instrument_master)
    my.calculate_ars_forecast('2020-06-13', '2020-07-13', 30, True, True, True, False)
