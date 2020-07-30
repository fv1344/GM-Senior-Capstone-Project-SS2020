# Load other local Python modules to be used in this MAIN module
from SourceFolder.PythonProjectFolder.DataFetch import DataFetch
from SourceFolder.PythonProjectFolder.DataForecast import DataForecast
from SourceFolder.PythonProjectFolder.dbEngine import DBEngine
from SourceFolder.PythonProjectFolder.BuySell import BuySell
from SourceFolder.PythonProjectFolder.EngineeredFeatures import EngineeredFeatures
from SourceFolder.PythonProjectFolder.TradingSimulator import TradingSimulator
import SourceFolder.PythonProjectFolder.AccuracyTest

# instrument symbol table
instrument_master = 'dbo_instrumentmaster'

"""
    THESE PHASES ARE THE ORDER TO RUN THE APPLICATION FOR PROPER INITIAL POPULATION
    EACH SUCCESSIVE RUN CAN BE CUSTOMIZED USING THE BOOLEAN VARIABLES BELOW
    Phase 1: Populate the date dimension
    Phase 2: Collect the source data
    Phase 3: Calculate engineered data from close prices
    Phase 4: Forecast close prices
    Phase 5: Generate Buy/Sell signals
    Phase 6: Run investment simulation
"""

"""
    TOGGLE THE FOLLOWING BOOLEANS TO RUN THE DESIRED PORTIONS OF THE APPLICATION
    COMMENTED BOOLEANS REPRESENT PORTIONS OF RUNTIME INTENSIVE FUNCTIONS (RUNTIME WILL VARY BETWEEN MACHINES)
"""

# Populate the date dimension
reset_date_dim = False                  # (Takes around 1 minute)

# Collect the source data
update_close_stats = False
update_macro_stats = False

# Calculate engineered data from close prices
update_engineered_features = False

# Forecast close prices
update_msf_forecast = False             # (Takes around 3 minutes)
update_ars_forecast = False
update_fjf_forecast = False             # (Takes around 5 minutes)
update_lr_forecast = False              # (Takes around 2 minutes)
update_remaining_forecasts = False      # (Takes around 1 hour)

# Generate Buy/Sell signals
update_signals = False                  # (Takes around 5-10 minutes)

# Run investment simulation
run_simulator = False                   # (Takes around 15 minutes)

"""
    OPERATIONS BELOW
"""

# Create database connection
db_engine = DBEngine().mysql_engine()

# Get instrument data
if update_close_stats:
    print("Getting Instrument Close Prices...")
    # Get raw market data
    master_data = DataFetch(db_engine, instrument_master)
    # Get ticker symbols
    ticker_symbols = master_data.get_datasources()
    # Get data from Yahoo! Finance and store in InstrumentStatistics
    master_data.get_data(ticker_symbols)

# Get date data and store in DateDim
if reset_date_dim:
    print("Populating The Date Dimension...")
    reset_calender = DataFetch(db_engine, instrument_master)
    reset_calender.get_calendar("2000-01-01", "2025-12-31")

# Get Macroeconomic indicator data
if update_macro_stats:
    print("Getting Macroeconomic Indicator Statistics...")
    SourceFolder.PythonProjectFolder.AccuracyTest.get_past_data(db_engine)
    DataFetch.macroFetch(db_engine)

# Generate MSF forecasts
if update_msf_forecast:
    print("Generating MSF Forecasts...")
    # DataForecast.MSF_final(db_engine)
    DataForecast.MSF1(db_engine)
    DataForecast.MSF2(db_engine)
    DataForecast.MSF3(db_engine)
    DataForecast.MSF2_Past_Date(db_engine)

# Generate data engineered from close prices
if update_engineered_features:
    print("Calculating Engineered Features...")
    # Get raw data from database to calculate forecasts
    indicators = EngineeredFeatures(db_engine, instrument_master)
    # Calculate technical indicators and store in EngineeredFeatures
    indicators.calculate()

# Generate the majority of the forecasts
if update_remaining_forecasts:
    print("Calculating...")
    # Get raw data from database to calculate forecasts
    forecast = DataForecast(db_engine, instrument_master)

    # Polynomial regression function
    # Takes a while to run, comment out if need be
    # Params: start regression analysis date, number of days to use for analysis, forecast amount of days
    # Flawed implementation because every record is deleted and recomputed every run
    # 1 year = 1.5 min runtime
    # Each run will take the same amount of time
    print("Polynomial Regression...")
    forecast.calculate_regression("2019-07-30", 20, 5)

    # calculate and store price predictions ("PricePred")
    # 10 years = 2 min runtime
    # Coded to not overwrite whole table every time (i.e. each run after first will be quick)
    print("'Price Prediction'...")
    forecast.calculate_forecast()

    # flawed price prediction from previous semesters, without our improvements ("PricePredOld")
    # 10 years = 2 min runtime
    # Coded to not overwrite whole table every time (i.e. each run after first will be quick)
    print("'Price Prediction Old'...")
    forecast.calculate_forecast_old()

    # calculate and store ARIMA forecast
    # 10 years = 3 min runtime
    # Coded to not overwrite whole table every time (i.e. each run after first will be quick)
    print("ARIMA...")
    forecast.calculate_arima_forecast()

    # calculate and store Random Forest forecast
    # 1 year = 3.5min runtime
    # Coded to not overwrite whole table every time (i.e. each run after first will be quick)
    print("Random Forest...")
    forecast.calculate_random_forest_forecast("2019-07-30")

    # calculate and store SVM forecast
    # 10 years = 2.5 min runtime
    # Coded to not overwrite whole table every time (i.e. each run after first will be quick)
    print("SVM...")
    forecast.calculate_svm_forecast()

    # calculate and store XGBoost forecast
    # 10 years = 3.5 min runtime
    # Coded to not overwrite whole table every time (i.e. each run after first will be quick)
    print("XG Boost...")
    forecast.calculate_xgboost_forecast()

# Generate Buy / Sell Signals
if update_signals:
    print("Generating Buy/Sell Signals...")
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

# Generate investment simulation results
if run_simulator:
    print("Running Investment Simulator...")
    # Run Trade Simulations Based on Trade Signals
    simulator = TradingSimulator(db_engine, instrument_master)
    # individual strategies
    simulator.trade_sim()
    # combination trading strategy
    simulator.combination_trade_sim()
    # buy and hold simulation
    simulator.buy_hold_sim()

# Generate Aman Range Shift forecasts
if update_ars_forecast:
    print("Calculating ARS...")
    my = DataForecast(db_engine, instrument_master)
    my.calculate_ars_forecast('2020-07-30', '2020-08-30', 30, False, True, False, False)

# Generate Frino Jais Function forecasts
if update_fjf_forecast:
    print("Calculating FJF...")
    my = DataForecast(db_engine, instrument_master)
    my.FJF()

# Generate Linear Regression forecasts
if update_lr_forecast:
    print("Calculating Linear Regression...")
    # Get raw data from database to calculate forecasts
    forecast = DataForecast(db_engine, instrument_master)
    forecast.calculate_lr_forecast()
