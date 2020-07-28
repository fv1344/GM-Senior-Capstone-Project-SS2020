# import libraries to be used in this code module
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from math import sqrt
from statistics import stdev
import numpy as np
import xgboost as xgb
import calendar
import datetime as dt
from datetime import timedelta, datetime
import FinsterTab.W2020.AccuracyTest
import sqlalchemy as sal
from sklearn.model_selection import train_test_split    # not used at this time
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import random as rand
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
#import keras
#from keras.models import Sequential
#from keras.layers import Dense, LSTM
#from tensorflow.keras import Sequential
from sklearn.preprocessing import *
import math


# class declaration and definition
from FinsterTab.W2020.dbEngine import DBEngine


class DataForecast:
    def __init__(self, engine, table_name):
        """
        Calculate historic one day returns and 10 days of future price forecast
        based on various methods
        Store results in dbo_AlgorithmForecast table
        :param engine: provides connection to MySQL Server
        :param table_name: table name where ticker symbols are stored
        """
        self.engine = engine
        self.table_name = table_name

    def FJF(self):

        '''Frino Jais Function(FJF) for stock market forecasting'''
        '''Uses Deep Learning and Neural Networks for forecast using the previous 90 days of close prices for each stock'''
        query = 'SELECT * FROM {}'.format(self.table_name)
        df = pd.read_sql_query(query, self.engine)

        algoCode = "'FJF'"  # Master `algocode` for improved prediction from previous group, user created codes

        # add code to database if it doesn't exist
        code_query = 'SELECT COUNT(*) FROM dbo_algorithmmaster WHERE algorithmcode=%s' % algoCode
        count = pd.read_sql_query(code_query, self.engine)
        if count.iat[0, 0] == 0:
            algoName = "'FJF'"
            insert_code_query = 'INSERT INTO dbo_algorithmmaster VALUES({},{})'.format(algoCode, algoName)
            self.engine.execute(insert_code_query)

        # loop through each ticker symbol
        for ID in df['instrumentid']:
            # remove all future prediction dates - these need to be recalculated daily
            remove_future_query = 'DELETE FROM dbo_algorithmforecast WHERE algorithmcode={} AND prederror=0 AND ' \
                                  'instrumentID={}'.format(algoCode, ID)
            self.engine.execute(remove_future_query)

            # find the latest forecast date
            date_query = 'SELECT ForecastDate FROM dbo_AlgorithmForecast WHERE AlgorithmCode={} AND InstrumentID={} ' \
                         'ORDER BY ForecastDate DESC LIMIT 1'.format(algoCode, ID)
            latest_date = pd.read_sql_query(date_query, self.engine)  # most recent forecast date calculation

            # if table has forecast prices already find the latest one and delete it
            # need to use most recent data for today if before market close at 4pm
            if not latest_date.empty:
                latest_date_str = "'" + str(latest_date['ForecastDate'][0]) + "'"
                delete_query = 'DELETE FROM dbo_algorithmforecast WHERE algorithmcode={} AND instrumentid={} AND ' \
                               'forecastdate={}'.format(algoCode, ID, latest_date_str)
                self.engine.execute(delete_query)

            # get raw price data from database
            data_query = 'SELECT date, close FROM dbo_instrumentstatistics WHERE instrumentid=%s ORDER BY date ASC' % ID
            data = pd.read_sql_query(data_query, self.engine) # close values in dataframe

            # if else to make writing stock names easier
            if (ID==1):
                stockName="GM"
            elif (ID==2):
                stockName="PFE"
            elif (ID==3):
                stockName="SPY"
            elif (ID==4):
                stockName="XPH"
            elif (ID==5):
                stockName="CARZ"
            else:
                stockName="^TYZ"

            # i want to get just the close values from the instrument stats table
            closeData = data.filter(['close'])
            # these close values will be converted into a numpy 2D array for easier processing
            closeDataSet = closeData.values
            # would like to train 80% of data, standard training rate and round up if it is a decimal
            trainingLength = math.ceil(len(closeDataSet) * 0.8) # for the 80:20 training:testing model

            #Scale the data, this is a method that must be used when using neural networks
            #input data is being preprocessed
            dataScaler = MinMaxScaler(feature_range=(0,1)) # data will be scaled in values from 0 to 1 which is easier to be read by keras
            scaledData = dataScaler.fit_transform(closeDataSet) # transform close data set to compute the min and max values to be used to scaling

            #Create training data set
            trainingDataSet = scaledData[0:trainingLength, :] #create scaled training data set which will contain the values from index 0 to training length

            pastTrain=[] #independent training variables, these will be the stored close values used to train the model
            futureTrain=[] #dependent training varaibles, this array will store the prediction

            #Create a for loop to pull from the last 90 days
            #every time the for loop is ran, a prediction value will be stored into future train until the end of the training data set length is reached
            for i in range (90, len(trainingDataSet)): #only looking at training data set now
                pastTrain.append(trainingDataSet[i-90:i,0]) # store the past scaled values from index 0 to 89
                futureTrain.append(trainingDataSet[i, 0]) # will contain 91st values indexed at 90 this will be forecast values to be trained

            pastTrain=np.array(pastTrain) # array that hold the past 90 close prices
            futureTrain=np.array(futureTrain) # will contain the "perfect forecast price"
            #print(pastTrain.shape)
            # Data must be reshaped because keras requires a 3-d model
            pastTrain=np.reshape(pastTrain, (pastTrain.shape[0], 90, 1)) # (Number of samples, number of timesteps, number of features)

            #LSTM MODEL
            #Have to build layers for the LSTM model
            lstmModel =Sequential() # the sequential class in keras takes input and produces outputs based on layers
            lstmModel.add(LSTM(50, return_sequences=True, input_shape=(90,1))) # giving the first layer our past 90 day input
            lstmModel.add(LSTM(50, return_sequences=False)) # rule of thumb to follow: sample size/2
            lstmModel.add(Dense(30)) #outputs a shape of 25 # giving outpupt predictions
            lstmModel.add(Dense(1)) #output a shape of 1

            # Declare optimizer and loss function
            #optimizer adaptive moment estimation
            lstmModel.compile(optimizer = 'adam', loss = 'mean_squared_error') # optimizer used to improve loss function and loss function is used to test accuracy

            #Training
            # batch size is the group of data
            # epoch is how many times the from start to end
            # stochastic gradient descent
            print("Learning data for "+ stockName+". Please wait...")
            lstmModel.fit(pastTrain,futureTrain, batch_size=1, epochs=1)

            #training is done now

            #Create testing data  (want to try to predict the last 20% of data)
            testData=scaledData[trainingLength-90:,:] # this array will contain values from the current row number 1118 to row 1208 (90 days prediction)

            pastTest=[] #past 90 values
            futureTest=closeDataSet[trainingLength:,:] #prediction values stored here from 1208 onwards

            for i in range(90, len(testData)):
                pastTest.append(testData[i-90:i, 0]) #store the last 90 values to pastTest


            pastTest=np.array(pastTest) #convert to numpy array

            pastTest=np.reshape(pastTest,(pastTest.shape[0],90,1)) #reshape xTest to 3-d format so it is easier to be read by keras

            forecastPrices =lstmModel.predict(pastTest) # forecast prices based on past test values (last 90 days)
            forecastPrices = dataScaler.inverse_transform(forecastPrices)

            # Root mean squared error test
            RMSE = np.sqrt(np.mean((forecastPrices - futureTest) ** 2))

            print("")
            print("The root mean squared error for the model's prediction to actual close values is: " + str(RMSE))
            print("")

            '''print("")
            print("The mean squared error for " +stockName+" is: "+str(RMSE))

            print("Forecast prices: ")
            forecastDf = pd.DataFrame(data=forecastPrices)
            print(forecastDf)
            print("Y test: ")
            print(yTest)

            print("These are forecast prices over the last " + str(len(forecastPrices)) + " days")'''

            #trying something different
            last90days = closeData[-90:].values # close data from the last 90 days

            scaledLast90Days = dataScaler.transform(last90days)
            independentTest=[]
            independentTest.append(scaledLast90Days)
            independentTest=np.array(independentTest)
            independentTest=np.reshape(independentTest,(independentTest.shape[0],independentTest.shape[1],1))
            predPrice=lstmModel.predict(independentTest)
            predPrice=dataScaler.inverse_transform(predPrice)
            print("")
            print("The estimated stock price for " + stockName +" on ")
            tomorrowDate =dt.date.today() + dt.timedelta(days=1)
            print(dt.date.today()+dt.timedelta(days=1))
            print("is: " +str(predPrice))
            print("")






    def FJF2(self):

        '''Frino Jais Function(FJF) for stock market forecasting'''
        '''Uses Deep Learning and Neural Networks for forecast using the previous 90 days of close prices for each stock'''
        query = 'SELECT * FROM {}'.format(self.table_name)
        df = pd.read_sql_query(query, self.engine)

        algoCode = "'FJF'"  # Master `algocode` for improved prediction from previous group, user created codes

        # add code to database if it doesn't exist
        code_query = 'SELECT COUNT(*) FROM dbo_algorithmmaster WHERE algorithmcode=%s' % algoCode
        count = pd.read_sql_query(code_query, self.engine)
        if count.iat[0, 0] == 0:
            algoName = "'FrinoJaisFunction'"
            insert_code_query = 'INSERT INTO dbo_algorithmmaster VALUES({},{})'.format(algoCode, algoName)
            self.engine.execute(insert_code_query)

        # loop through each ticker symbol
        for ID in df['instrumentid']:
            # remove all future prediction dates - these need to be recalculated daily
            remove_future_query = 'DELETE FROM dbo_algorithmforecast WHERE algorithmcode={} AND prederror=0 AND ' \
                                  'instrumentID={}'.format(algoCode, ID)
            self.engine.execute(remove_future_query)

            # find the latest forecast date
            date_query = 'SELECT ForecastDate FROM dbo_AlgorithmForecast WHERE AlgorithmCode={} AND InstrumentID={} ' \
                         'ORDER BY ForecastDate DESC LIMIT 1'.format(algoCode, ID)
            latest_date = pd.read_sql_query(date_query, self.engine)  # most recent forecast date calculation

            # if table has forecast prices already find the latest one and delete it
            # need to use most recent data for today if before market close at 4pm
            if not latest_date.empty:
                latest_date_str = "'" + str(latest_date['ForecastDate'][0]) + "'"
                delete_query = 'DELETE FROM dbo_algorithmforecast WHERE algorithmcode={} AND instrumentid={} AND ' \
                               'forecastdate={}'.format(algoCode, ID, latest_date_str)
                self.engine.execute(delete_query)

            # get raw price data from database
            data_query = 'SELECT date, close FROM dbo_instrumentstatistics WHERE instrumentid=%s ORDER BY date ASC' % ID
            data = pd.read_sql_query(data_query, self.engine) # close values in dataframe

            # if else to make writing stock names easier
            if (ID==1):
                stockName="GM"
            elif (ID==2):
                stockName="PFE"
            elif (ID==3):
                stockName="SPY"
            elif (ID==4):
                stockName="XPH"
            elif (ID==5):
                stockName="CARZ"
            else:
                stockName="^TYZ"

            # i want to get just the close values from the instrument stats table
            closeData = data.filter(['close'])
            # these close values will be converted into a numpy 2D array for easier processing
            closeDataSet = closeData.values
            # would like to train 80% of data, standard training rate and round up if it is a decimal
            trainingLength = math.ceil(len(closeDataSet) * 0.8) # for the 80:20 training:testing model

            #Scale the data, this is a method that must be used when using neural networks
            #input data is being preprocessed
            dataScaler = MinMaxScaler(feature_range=(0,1)) # data will be scaled in values from 0 to 1 which is easier to be read by keras
            scaledData = dataScaler.fit_transform(closeDataSet) # transform close data set to compute the min and max values to be used to scaling

            #Create training data set
            trainingDataSet = scaledData[0:trainingLength, :] #create scaled training data set which will contain the values from index 0 to training length

            pastTrain=[] #independent training variables, these will be the stored close values used to train the model
            futureTrain=[] #dependent training varaibles, this array will store the prediction

            #Create a for loop to pull from the last 90 days
            #every time the for loop is ran, a prediction value will be stored into future train until the end of the training data set length is reached
            for i in range (90, len(trainingDataSet)): #only looking at training data set now
                pastTrain.append(trainingDataSet[i-90:i,0]) # store the past scaled values from index 0 to 89
                futureTrain.append(trainingDataSet[i, 0]) # will contain 91st values indexed at 90 this will be forecast values to be trained

            pastTrain=np.array(pastTrain) # array that hold the past 90 close prices
            futureTrain=np.array(futureTrain) # will contain the "perfect forecast price"
            print(pastTrain.shape)
            # Data must be reshaped because keras requires a 3-d model
            pastTrain=np.reshape(pastTrain, (pastTrain.shape[0], 90, 1)) # (Number of samples, number of timesteps, number of features)

            #LSTM MODEL
            #Have to build layers for the LSTM model
            lstmModel =Sequential() # the sequential class in keras takes input and produces outputs based on layers
            lstmModel.add(LSTM(50, return_sequences=True, input_shape=(90,1))) # giving the first layer our past 90 day input
            lstmModel.add(LSTM(50, return_sequences=False)) # rule of thumb to follow: sample size/2
            lstmModel.add(Dense(30)) #outputs a shape of 25 # giving outpupt predictions
            lstmModel.add(Dense(1)) #output a shape of 1

            # Declare optimizer and loss function
            #optimizer adaptive moment estimation
            lstmModel.compile(optimizer = 'adam', loss = 'mean_squared_error') # optimizer used to improve loss function and loss function is used to test accuracy

            #Training
            # batch size is the group of data
            # epoch is how many times the from start to end
            # stochastic gradient descent
            print("Learning data for "+ stockName+". Please wait...")
            lstmModel.fit(pastTrain,futureTrain, batch_size=1, epochs=1)

            #training is done now

            #Create testing data  (want to try to predict the last 20% of data)
            testData=scaledData[trainingLength-90:,:] # this array will contain values from the current row number 1118 to row 1208 (90 days prediction)

            pastTest=[] #past 90 values
            futureTest=closeDataSet[trainingLength:,:] #prediction values stored here from 1208 onwards

            for i in range(90, len(testData)):
                pastTest.append(testData[i-90:i, 0]) #store the last 90 values to pastTest


            pastTest=np.array(pastTest) #convert to numpy array

            pastTest=np.reshape(pastTest,(pastTest.shape[0],90,1)) #reshape xTest to 3-d format so it is easier to be read by keras

            forecastPrices =lstmModel.predict(pastTest) # forecast prices based on past test values (last 90 days)
            forecastPrices = dataScaler.inverse_transform(forecastPrices)

            # Root mean squared error test
            RMSE = np.sqrt(np.mean((forecastPrices - futureTest) ** 2))

            print("")
            print("The root mean squared error for the model's prediction to actual close values is: " + str(RMSE))
            print("")

            last90days = closeData[-90:].values  # close data from the last 90 days

            futureForecast=[]
            fcDate=datetime.now()
            i=0

            #loop to output next 30 days (WORKS)
            while (i<30):
                scaledLast90Days = dataScaler.transform(last90days)  # transform he values to be between 0 and 1
                independentTest = []  # create an empty list to store last 90 days
                independentTest.append(scaledLast90Days)  # store the last 90 days of scaled values
                independentTest = np.array(independentTest)  # change the list in to an array
                independentTest = np.reshape(independentTest,(independentTest.shape[0], independentTest.shape[1], 1))  # 2d to 3d
                nextDayForecast = lstmModel.predict(independentTest)  # get prediction value for next day
                nextDayForecast = dataScaler.inverse_transform(nextDayForecast)  # transform scaled value to normal format
                futureForecast.extend(nextDayForecast.tolist())  # add new value to future forecast list
                last90days=np.append(last90days,nextDayForecast[0]) # add new values to 90 day set
                last90days=last90days[1:] # move one index up to the previous 90 values that now include the forecast price to predict the next price
                last90days=last90days.reshape(-1,1) #reshape to 2d

                insert_query = 'INSERT INTO dbo_algorithmforecast VALUES ("{}",{},{},{},{})'
                forecastClose=nextDayForecast
                predError=0
                fcDate=fcDate + timedelta(days=1)
                forecastDate=fcDate.strftime("%Y-%m-%d")
                forecastClose = forecastClose.flatten() # have to pull the value out of the 2d array
                insert_query=insert_query.format(forecastDate,ID,forecastClose[0],algoCode,predError)
                self.engine.execute(insert_query)


                i=i+1



            '''
                    if(len(tempInput)>90): # after else statement tempInput will be of size 91
                    last90days=np.array(tempInput[1:]) # we will shift one index up to include the forecasted price in the next prediction
                    print("{} day input {}".format(i,last90days))
                    last90days=last90days.reshape(1,-1) #reshape to 2d
                    last90days= last90days.reshape(1,90,1) #reshape to 3d
                    nextDayForecast=lstmModel.predict(last90days) # get next day forecast
                    print("{} day output {}".format(i,nextDayForecast))
                    tempInput.extend(nextDayForecast[0].tolist()) # add forecasted value to acting 90 day set
                    tempInput=tempInput[1:] #move one index forward
                    futureForecast.extend(nextDayForecast.tolist()) #add the next forecasted value to the forecast set
                    i=i+1
                else:
                    scaledLast90Days = dataScaler.transform(last90days) #transform he values to be between 0 and 1
                    independentTest=[] #create an empty list to store last 90 days
                    independentTest.append(scaledLast90Days) #store the last 90 days of scaled values
                    independentTest=np.array(independentTest) # change the list in to an array
                    independentTest=np.reshape(independentTest,(independentTest.shape[0],independentTest.shape[1],1)) # 2d to 3d
                    nextDayForecast=lstmModel.predict(independentTest) # get prediction value for next day
                    nextDayForecast=dataScaler.inverse_transform(nextDayForecast) # transform scaled value to normal format
                    tempInput.extend(nextDayForecast) # add new value to existing 90 ay data set
                    futureForecast.extend(nextDayForecast.tolist()) # add new value to future forecast list
                    i=i+1
                    '''


            '''
                    last90days=last90days.reshape(1,90,1)
                    nextDayForecast=lstmModel.predict(last90days) #get the next day forecast using previous 90 values
                    print(nextDayForecast[0]) # why is this showing a weird prediction?
                    tempInput.extend(nextDayForecast[0].tolist())# store the forecasted value into our acting 90 day set
                    print(len(tempInput))
                    futureForecast.extend(nextDayForecast.tolist()) # store the forecasted value into its own set
                    i=i+1
                    '''

            print("Future forecast is: ")
            print(futureForecast)
            print(len(futureForecast))


    def calculate_william_forecast3(self):

        # retrieve InstrumentsMaster table from database
        query = 'SELECT * FROM {}'.format(self.table_name)
        df = pd.read_sql_query(query, self.engine)
        algoCode = "'ARS'"

        # add code to database if it doesn't exist
        code_query = 'SELECT COUNT(*) FROM dbo_algorithmmaster WHERE algorithmcode=%s' % algoCode
        count = pd.read_sql_query(code_query, self.engine)
        if count.iat[0, 0] == 0:
            algoName = "'AmanRangeShift'"
            insert_code_query = 'INSERT INTO dbo_algorithmmaster VALUES({},{})'.format(algoCode, algoName)
            self.engine.execute(insert_code_query)

        # Number of past values to use to influence each forecast
        num_analyze_days = 30

        # Number of close prices to forecast
        num_forecast_days = 30

        # Loop through each instrument one at a time
        # This outer loop section is visited once for each instrument, before their first forecast
        for ID in df['instrumentid']:

            # SET PARAMETERS BEFORE EACH INSTRUMENT'S FIRST FORECAST

            # Get "num_analyze_days" amount of real closing values and save into "close_and_date_data"
            query = 'SELECT date, close FROM dbo_instrumentstatistics WHERE instrumentid={} ' \
                    'ORDER BY Date DESC LIMIT {}'.format(ID, num_analyze_days)
            close_and_date_data = pd.read_sql_query(query, self.engine)

            # Save "global" values for each instrument to use
            real_max = close_and_date_data['close'].max()
            real_min = close_and_date_data['close'].min()
            real_avg = close_and_date_data['close'].mean()
            real_range = real_max - real_min
            real_max_deviation = real_range / real_avg
            print("Global Max: ${}".format(real_max))
            print("Global Min: ${}".format(real_min))
            print("Global Avg: ${:.2f}".format(real_avg))
            print("Global Range: ${:.2f}".format(real_range))
            print("Global Max Deviation: {:.2f}%".format(real_max_deviation*100))

            # Set influence values to default value at the beginning od each instruments forecast cycle
            influence_up = False
            influence_down = False

            # Get the most recent market close date
            last_close_date = close_and_date_data['date'].iloc[0]

            # Collect "num_forecast_days" amount of dates and save into "future_dates".
            query = 'SELECT date FROM dbo_datedim WHERE date > "{}" AND weekend = 0 AND isholiday = 0 ORDER BY Date ASC LIMIT {}'.format(
                last_close_date, num_forecast_days)
            future_dates = pd.read_sql_query(query, self.engine)

            # Loop through the future, 1 day at a time (day stores the date)
            for day in future_dates['date']:

                # PERFORM CALCULATIONS
                # Newest forecasts are inserted into the front, so most recent close is at index 0
                last_close = close_and_date_data['close'].iloc[0]
                max_close = close_and_date_data['close'].iloc[:num_analyze_days].max()
                min_close = close_and_date_data['close'].iloc[:num_analyze_days].min()
                avg_close = close_and_date_data['close'].iloc[:num_analyze_days].mean()
                # The amount of dollars the forecast is allowed to range between
                forecast_price_range = last_close * real_max_deviation
                # The neutral limits of where the forecast can land between
                neutral_lower_range = last_close - (forecast_price_range / 2)
                neutral_upper_range = last_close + (forecast_price_range / 2)

                # Shift Forecast Range
                # If the last close price is trending lower than the average
                if last_close < avg_close:
                    ascend = False
                    # How severe is it down
                    down_percent = (avg_close - last_close) / avg_close
                    # Adjust the forecast_price_range to the same percent as the trend
                    shift_amount = forecast_price_range * down_percent
                    shifted_lower_range = neutral_lower_range - shift_amount
                    shifted_upper_range = neutral_upper_range - shift_amount
                # If the last close price is trending higher than the average
                else:
                    ascend = True
                    # How severe is it up
                    up_percent = (last_close - avg_close) / avg_close
                    # Adjust the forecast_price_range to the same percent as the trend
                    shift_amount = forecast_price_range * up_percent
                    shifted_lower_range = neutral_lower_range + shift_amount
                    shifted_upper_range = neutral_upper_range + shift_amount

                # print results
                print("\n\n***** {}-Day Close Statistics for {} *****\n".format(num_analyze_days, ID))
                # print(close_and_date_data)
                print("\nMaximum: ${:.2f}".format(max_close))
                print("Minimum: ${:.2f}".format(min_close))
                print("Current: ${:.2f}".format(last_close))
                print("Average: ${:.2f}".format(avg_close))

                print("\n\n***** Forecast Range *****")
                print("Maximum Deviation Constant: {:.2f}%".format(real_max_deviation * 100))
                print("{:.2f}% of ${:.2f} = ${:.2f} of forecast range"
                      .format(real_max_deviation * 100, last_close, forecast_price_range))
                print("Neutral Forecast Range: ${:.2f} - ${:.2f}".format(neutral_lower_range, neutral_upper_range))

                print("\n\n***** Shift Forecast Range *****")
                if ascend:
                    print("Current close is {:.2f}% higher than the average".format(up_percent * 100))
                    print("Shifting the forecast range in favor of an increase: {:.2f}% of ${:.2f} = ${:.2f}"
                          .format(up_percent * 100, forecast_price_range, shift_amount))
                else:
                    print("Current close is {:.2f}% lower than the average".format(down_percent * 100))
                    print("Shifting the forecast range in favor of a decrease: {:.2f}% of ${:.2f} = ${:.2f}"
                          .format(down_percent * 100, forecast_price_range, shift_amount))
                print("Shifted Forecast Range: ${:.2f} - ${:.2f}".format(shifted_lower_range, shifted_upper_range))

                # Generate Forecasts
                print("\n\n***** Generate Forecast Close Price *****")
                forecast_choice_average = (shifted_upper_range + shifted_lower_range) / 2
                rand.seed(datetime.now())
                # The last forecast was below the 30 day minimum. Force an increase
                if influence_up:
                    forecast_choice_random = rand.uniform(last_close, last_close + forecast_price_range)
                    print("Forced Increase")
                # The last forecast was above the 30 day maximum. Force a decrease
                elif influence_down:
                    forecast_choice_random = rand.uniform(shifted_lower_range - forecast_price_range, last_close)
                    print("Forced Decrease")
                else:
                    forecast_choice_random = rand.uniform(shifted_lower_range, shifted_upper_range)
                    print("Regular shifted range")
                print("Forecasted Close Price for 07/01/2020 (Average in shifted range): ${:.2f}".format(
                    forecast_choice_average))
                print("Forecasted Close Price for 07/01/2020 (Random in shifted range): ${:.2f}".format(
                    forecast_choice_random))

                # Adjust next run based on forecast
                if forecast_choice_random < real_min:
                    influence_up = True
                    print("Next will be forced increase")
                elif forecast_choice_random > real_max:
                    influence_down = True
                    print("Next will be forced decrease")
                else:
                    influence_up = False
                    influence_down = False
                    print("Next will be normal")

                print("________________________")

                # Append new forecast to beginning of "close_date_and_data"
                new_forecast = pd.DataFrame({'date': day, 'close': forecast_choice_random.__round__(2)}, index=[0])
                close_and_date_data = pd.concat([new_forecast, close_and_date_data]).reset_index(drop=True)
                print(close_and_date_data.iloc[0:30])

                # insert into database
                insert_query = 'INSERT INTO dbo_algorithmforecast VALUES ("{}", {}, {}, {}, {})'
                forecastClose = forecast_choice_random.__round__(2)
                predError = 0
                forecastDate = day
                insert_query = insert_query.format(forecastDate, ID, forecastClose, algoCode, predError)
                self.engine.execute(insert_query)

    """
        PROBLEM: Allowing the maximum deviation to be influenced by the forecasted values means that the percentage will
        continue to grow as the limits are reached. Which means the forecast range allowance will also grow, 
        and this will continue in a cycle of expanding limits.
        
        PROPOSED SOLUTION: Use the maximum deviation of the historical close prices as a constant, not a variable. This
        will still allow the forecasts to fluctuate, but they will fluctuate in a range that has been proven to happen
        and nothing more. To add more variance, this constant can be scaled.
    """

    def calculate_william_forecast2(self):

        # Get list of ticker symbols
        pd.set_option('mode.chained_assignment', None)
        query = 'SELECT * FROM %s' % self.table_name
        df = pd.read_sql_query(query, self.engine)
        algoCode = "'ARS'"

        # Number of past values to use to influence each forecast
        num_analyze_days = 30

        # Number of close prices to forecast
        num_forecast_days = 30

        # Loop through each instrument one at a time
        # This outer loop section is visited once for each instrument, before their first forecast
        for ID in df['instrumentid']:

            # SET PARAMETERS BEFORE EACH INSTRUMENT'S FIRST FORECAST

            # Get "num_analyze_days" amount of real closing values and save into "close_and_date_data"
            query = 'SELECT date, close FROM dbo_instrumentstatistics WHERE instrumentid={} ' \
                    'ORDER BY Date DESC LIMIT {}'.format(ID, num_analyze_days)
            close_and_date_data = pd.read_sql_query(query, self.engine)

            # save real max and min values for use in "Influence Forecast" section
            real_max = close_and_date_data['close'].max()
            real_min = close_and_date_data['close'].min()
            print("Global Max: {}".format(real_max))
            print("Global Min: {}".format(real_min))

            # Set influence values to default value at the beginning od each instruments forecast cycle
            influence_up = False
            influence_down = False

            # Get the most recent market close date
            last_close_date = close_and_date_data['date'].iloc[0]

            # Collect "num_forecast_days" amount of dates and save into "future_dates".
            query = 'SELECT date FROM dbo_datedim WHERE date > "{}" AND weekend = 0 AND isholiday = 0 ORDER BY Date ASC LIMIT {}'.format(
                last_close_date, 30)
            future_dates = pd.read_sql_query(query, self.engine)

            # Loop through the future, 1 day at a time (day stores the date)
            for day in future_dates['date']:

                # PERFORM CALCULATIONS
                # Newest forecasts are inserted into the front, so most recent close is at index 0
                last_close = close_and_date_data['close'].iloc[0]
                # Only want to use the most recent 30 days for calculations. iloc range upper limit is excluded
                max_close = close_and_date_data['close'].iloc[:num_analyze_days].max()
                min_close = close_and_date_data['close'].iloc[:num_analyze_days].min()
                avg_close = close_and_date_data['close'].iloc[:num_analyze_days].mean()
                # The mathematical range of close prices
                diff_close = max_close - min_close
                # The maximum percent the close price varied from the average
                max_deviation = diff_close / avg_close
                # The amount of dollars the forecast is allowed to range between
                forecast_price_range = last_close * max_deviation
                # The neutral limits of where the forecast can land between
                neutral_lower_range = last_close - (forecast_price_range / 2)
                neutral_upper_range = last_close + (forecast_price_range / 2)

                # Shift Forecast Range
                # If the last close price is trending lower than the average
                if last_close < avg_close:
                    ascend = False
                    # How severe is it down
                    down_percent = (avg_close - last_close) / avg_close
                    # Adjust the forecast_price_range to the same percent as the trend
                    shift_amount = forecast_price_range * down_percent
                    shifted_lower_range = neutral_lower_range - shift_amount
                    shifted_upper_range = neutral_upper_range - shift_amount
                # If the last close price is trending higher than the average
                else:
                    ascend = True
                    # How severe is it up
                    up_percent = (last_close - avg_close) / avg_close
                    # Adjust the forecast_price_range to the same percent as the trend
                    shift_amount = forecast_price_range * up_percent
                    shifted_lower_range = neutral_lower_range + shift_amount
                    shifted_upper_range = neutral_upper_range + shift_amount

                # print results
                print("\n\n***** {}-Day Close Statistics for {} *****\n".format(num_analyze_days, ID))
                # print(close_and_date_data)
                print("\nMaximum: ${:.2f}".format(max_close))
                print("Minimum: ${:.2f}".format(min_close))
                print("Difference: ${:.2f}".format(diff_close))
                print("Current: ${:.2f}".format(last_close))
                print("Average: ${:.2f}".format(avg_close))

                print("\n\n***** Forecast Range *****")
                print("Maximum deviation from the average: {:.2f}%".format(max_deviation * 100))
                print("{:.2f}% of ${:.2f} = ${:.2f} of forecast range"
                      .format(max_deviation * 100, last_close, forecast_price_range))
                print("Neutral Forecast Range: ${:.2f} - ${:.2f}".format(neutral_lower_range, neutral_upper_range))

                print("\n\n***** Shift Forecast Range *****")
                if ascend:
                    print("Current close is {:.2f}% higher than the average".format(up_percent * 100))
                    print("Shifting the forecast range in favor of an increase: {:.2f}% of ${:.2f} = ${:.2f}"
                          .format(up_percent * 100, forecast_price_range, shift_amount))
                else:
                    print("Current close is {:.2f}% lower than the average".format(down_percent * 100))
                    print("Shifting the forecast range in favor of a decrease: {:.2f}% of ${:.2f} = ${:.2f}"
                          .format(down_percent * 100, forecast_price_range, shift_amount))
                print("Shifted Forecast Range: ${:.2f} - ${:.2f}".format(shifted_lower_range, shifted_upper_range))

                # Generate Forecasts
                print("\n\n***** Generate Forecast Close Price *****")
                forecast_choice_average = (shifted_upper_range + shifted_lower_range) / 2
                rand.seed(datetime.now())
                # The last forecast was below the 30 day minimum. Force an increase
                if influence_up:
                    forecast_choice_random = rand.uniform(last_close, last_close + forecast_price_range)
                    print("Forced Increase")
                # The last forecast was above the 30 day maximum. Force a decrease
                elif influence_down:
                    forecast_choice_random = rand.uniform(shifted_lower_range - forecast_price_range, last_close)
                    print("Forced Decrease")
                else:
                    forecast_choice_random = rand.uniform(shifted_lower_range, shifted_upper_range)
                    print("Regular shifed range")
                print("Forecasted Close Price for 07/01/2020 (Average in shifted range): ${:.2f}".format(
                    forecast_choice_average))
                print("Forecasted Close Price for 07/01/2020 (Random in shifted range): ${:.2f}".format(
                    forecast_choice_random))

                # Adjust next run based on forecast
                if forecast_choice_random < real_min:
                    influence_up = True
                    print("Next will be forced increase")
                elif forecast_choice_random > real_max:
                    influence_down = True
                    print("Next will be forced decrease")
                else:
                    influence_up = False
                    influence_down = False
                    print("Next will be normal")

                print("________________________")

                # Append new forecast to beginning of "close_date_and_data"
                new_forecast = pd.DataFrame({'date': day, 'close': forecast_choice_random.__round__(2)}, index=[0])
                close_and_date_data = pd.concat([new_forecast, close_and_date_data]).reset_index(drop=True)
                print(close_and_date_data.iloc[0:30])

                # insert into database
                insert_query = 'INSERT INTO dbo_algorithmforecast VALUES ("{}", {}, {}, {}, {})'
                forecastClose = forecast_choice_random.__round__(2)
                predError = 0
                forecastDate = day
                insert_query = insert_query.format(forecastDate, ID, forecastClose, algoCode, predError)
                self.engine.execute(insert_query)

        """
        Step 1: Find the volatility of the stock
        Step 2: Find the trend of the stock
        Step 3: Forecast a value
        Find the max and min close price over the past x days
        Find the SMA over the past x days
        Compare the difference of the max and min to the average stock price
         = Range/SMA
        This acts as the volatility of the stock
         - The higher the percent, the more volatile the stock
         - Ex: GM over 5 days
            - Max: $26.37
            - Min: $24.46
            - Range: $1.91
            - SMA: $25.518
            - 7.5% max deviation from the average
            - Current close = $24.46
            - 7.5% of $24.46 = $1.83 of potential change ($26.29 to 22.63)
            - Current close < Avg: Trending down at (25.518-24.46)/25.518 = 4% less than average
            - Tilt $1.83 4% more in favor of downward trend: $1.83*0.04 = $0.07 to shift
            - $26.29 - $0.07 = $26.22
            - $22.63 - $0.07 = $22.56
            - Average number in range = ($26.22 + $22.56)/2 = $24.39
            - Random number in range = $22.93


        Set those values as the limits of the forecast
        Find the simple moving average over the past x days
        Increase the limit of the value the SMA is trending towards
        """

    """
        PROBLEM: Due to the algorithm structure, specifically the shift component, the forecast will eventually approach
        zero or infinity. This is because the shift forecast direction is dependent on the last recorded close price
        compared to the average. So the first recorded close price, in comparison to the average, will dictate
        the absolute direction of all of the following forecasts.
        
        PROPOSED SOLUTION: Set upper and lower limits on whats allowed for a forecast. If a forecast exceeds a limit,
        force the next forecast to head the opposite direction, back into the playing field.
    """
    def calculate_william_forecast(self):

        # get list of ticker symbols
        pd.set_option('mode.chained_assignment', None)
        query = 'SELECT * FROM %s' % self.table_name
        df = pd.read_sql_query(query, self.engine)
        algoCode = "'ARS'"

        # Number of past values to pull
        close_range = 30

        # Number of close prices to forecast
        forecast_range = 30

        # Loop through each instrument 1 at a time
        for ID in df['instrumentid']:

            # get "close_range" amount of closing values and save into "close_and_date_data"
            query = 'SELECT date, close FROM dbo_instrumentstatistics WHERE instrumentid={} ' \
                    'ORDER BY Date DESC LIMIT {}'.format(ID, close_range)
            close_and_date_data = pd.read_sql_query(query, self.engine)

            # get the most recent market close date
            last_close_date = close_and_date_data['date'].iloc[0]

            # Collect "forecast_range" amount of dates and save into "future_dates". Include last close so graphs can be connected
            query = 'SELECT date FROM dbo_datedim WHERE date > "{}" AND weekend = 0 AND isholiday = 0 ORDER BY Date ASC LIMIT {}'.format(last_close_date, 30)
            future_dates = pd.read_sql_query(query, self.engine)


            # Loop through the future, 1 day at a time (day stores the date)
            for day in future_dates['date']:

                # perform calculations
                # Newest forecasts are inserted into the front, so most recent close is at index 0
                last_close = close_and_date_data['close'].iloc[0]
                # Only want to use the most recent 30 days for calculations. iloc range upper limit is excluded
                max_close = close_and_date_data['close'].iloc[:close_range].max()
                min_close = close_and_date_data['close'].iloc[:close_range].min()
                avg_close = close_and_date_data['close'].iloc[:close_range].mean()
                diff_close = max_close - min_close
                max_deviation = diff_close / avg_close
                forecast_range = last_close * max_deviation
                neutral_lower_range = last_close - (forecast_range / 2)
                neutral_upper_range = last_close + (forecast_range / 2)

                # Shift Forecast Range
                if last_close < avg_close:
                    ascend = False
                    down_percent = (avg_close-last_close)/avg_close
                    shift_amount = forecast_range*down_percent
                    shifted_lower_range = neutral_lower_range - shift_amount
                    shifted_upper_range = neutral_upper_range - shift_amount
                else:
                    ascend = True
                    up_percent = (last_close-avg_close)/avg_close
                    shift_amount = forecast_range * up_percent
                    shifted_lower_range = neutral_lower_range + shift_amount
                    shifted_upper_range = neutral_upper_range + shift_amount

                # print results
                print("\n\n***** {}-Day Close Statistics for {} *****\n".format(close_range, ID))
                # print(close_and_date_data)
                print("\nMaximum: ${:.2f}".format(max_close))
                print("Minimum: ${:.2f}".format(min_close))
                print("Difference: ${:.2f}".format(diff_close))
                print("Current: ${:.2f}".format(last_close))
                print("Average: ${:.2f}".format(avg_close))

                print("\n\n***** Forecast Range *****")
                print("Maximum deviation from the average: {:.2f}%".format(max_deviation * 100))
                print("{:.2f}% of ${:.2f} = ${:.2f} of forecast range"
                      .format(max_deviation * 100, last_close, forecast_range))
                print("Neutral Forecast Range: ${:.2f} - ${:.2f}".format(neutral_lower_range, neutral_upper_range))

                print("\n\n***** Shift Forecast Range *****")
                if ascend:
                    print("Current close is {:.2f}% higher than the average".format(up_percent * 100))
                    print("Shifting the forecast range in favor of an increase: {:.2f}% of ${:.2f} = ${:.2f}"
                          .format(up_percent * 100, forecast_range, shift_amount))
                else:
                    print("Current close is {:.2f}% lower than the average".format(down_percent * 100))
                    print("Shifting the forecast range in favor of a decrease: {:.2f}% of ${:.2f} = ${:.2f}"
                          .format(down_percent * 100, forecast_range, shift_amount))
                print("Shifted Forecast Range: ${:.2f} - ${:.2f}".format(shifted_lower_range, shifted_upper_range))

                # Generate Forecasts
                print("\n\n***** Generate Forecast Close Price *****")
                forecast_choice_average = (shifted_upper_range + shifted_lower_range) / 2
                rand.seed(datetime.now())
                forecast_choice_random = rand.uniform(shifted_lower_range, shifted_upper_range)
                print("Forecasted Close Price for 07/01/2020 (Average in shifted range): ${:.2f}".format(forecast_choice_average))
                print("Forecasted Close Price for 07/01/2020 (Random in shifted range): ${:.2f}".format(forecast_choice_random))

                print("________________________")


                # Append new forecast to beginning of "close_date_and_data"
                new_forecast = pd.DataFrame({'date': day, 'close': forecast_choice_random.__round__(2)}, index=[0])
                close_and_date_data = pd.concat([new_forecast, close_and_date_data]).reset_index(drop=True)
                print(close_and_date_data.iloc[0:30])


                # insert into database
                insert_query = 'INSERT INTO dbo_algorithmforecast VALUES ("{}", {}, {}, {}, {})'
                forecastClose = forecast_choice_random.__round__(2)
                predError = 0
                forecastDate = day
                insert_query = insert_query.format(forecastDate, ID, forecastClose, algoCode, predError)
                self.engine.execute(insert_query)

        """
        Step 1: Find the volatility of the stock
        Step 2: Find the trend of the stock
        Step 3: Forecast a value
        Find the max and min close price over the past x days
        Find the SMA over the past x days
        Compare the difference of the max and min to the average stock price
         = Range/SMA
        This acts as the volatility of the stock
         - The higher the percent, the more volatile the stock
         - Ex: GM over 5 days
            - Max: $26.37
            - Min: $24.46
            - Range: $1.91
            - SMA: $25.518
            - 7.5% max deviation from the average
            - Current close = $24.46
            - 7.5% of $24.46 = $1.83 of potential change ($26.29 to 22.63)
            - Current close < Avg: Trending down at (25.518-24.46)/25.518 = 4% less than average
            - Tilt $1.83 4% more in favor of downward trend: $1.83*0.04 = $0.07 to shift
            - $26.29 - $0.07 = $26.22
            - $22.63 - $0.07 = $22.56
            - Average number in range = ($26.22 + $22.56)/2 = $24.39
            - Random number in range = $22.93


        Set those values as the limits of the forecast
        Find the simple moving average over the past x days
        Increase the limit of the value the SMA is trending towards
        """


    def calculate_forecast(self):
        """
        Calculate historic one day returns based on traditional forecast model
        and 10 days of future price forecast
        Store results in dbo_AlgorithmForecast
        Improved forecast where we took out today's close price to predict today's price
        10 prior business days close prices are used as inputs to predict next day's price
        """

        # retrieve InstrumentMaster table from the database
        query = 'SELECT * FROM {}'.format(self.table_name)
        df = pd.read_sql_query(query, self.engine)
        algoCode = "'PricePred'"   # Master `algocode` for improved prediction from previous group, user created codes

        # add code to database if it doesn't exist
        code_query = 'SELECT COUNT(*) FROM dbo_algorithmmaster WHERE algorithmcode=%s' % algoCode
        count = pd.read_sql_query(code_query, self.engine)
        if count.iat[0, 0] == 0:
            algoName = "'PricePrediction'"
            insert_code_query = 'INSERT INTO dbo_algorithmmaster VALUES({},{})'.format(algoCode, algoName)
            self.engine.execute(insert_code_query)
        # loop through each ticker symbol
        for ID in df['instrumentid']:

            # remove all future prediction dates
            remove_future_query = 'DELETE FROM dbo_algorithmforecast WHERE algorithmcode={} AND prederror=0 AND ' \
                                  'instrumentid={}'.format(algoCode, ID)
            self.engine.execute(remove_future_query)

            # find the latest forecast date
            date_query = 'SELECT forecastdate FROM dbo_algorithmforecast WHERE algorithmcode={} AND instrumentid={} ' \
                         'ORDER BY forecastdate DESC LIMIT 1'.format(algoCode, ID)
            latest_date = pd.read_sql_query(date_query, self.engine) # most recent forecast date calculation

            # if table has forecast prices already find the latest one and delete it
            # need to use most recent data for today if before market close at 4pm
            if not latest_date.empty:
                latest_date_str = "'" + str(latest_date['forecastdate'][0]) + "'"
                delete_query = 'DELETE FROM dbo_algorithmforecast WHERE algorithmcode={} AND instrumentid={} AND ' \
                               'forecastdate={}'.format(algoCode, ID, latest_date_str)
                self.engine.execute(delete_query)

            # get raw price data from database
            data_query = 'SELECT A.date, A.close, B.ltrough, B.lpeak, B.lema, B.lcma, B.highfrllinelong, ' \
                         'B. medfrllinelong, B.lowfrllinelong FROM dbo_instrumentstatistics AS A, '\
                         'dbo_engineeredfeatures AS B WHERE A.instrumentid=B.instrumentid AND A.date=B.date ' \
                         'AND A.instrumentid=%s ORDER BY Date ASC' %ID
            data = pd.read_sql_query(data_query, self.engine)

            # prediction formula inputs
            # IF THESE VALUES ARE CHANGED, ALL RELATED PREDICTIONS STORED IN THE DATABASE BECOME INVALID!
            sMomentum = 2
            lMomentum = 5
            sDev = 10
            ma = 10
            start = max(sMomentum, lMomentum, sDev, ma)

            # calculate prediction inputs
            data['sMomentum'] = data['close'].diff(sMomentum)
            data['lMomentum'] = data['close'].diff(lMomentum)
            data['stDev'] = data['close'].rolling(sDev).std()
            data['movAvg'] = data['close'].rolling(ma).mean()

            # first predictions can be made after 'start' number of days
            for n in range(start, len(data)):
                insert_query = 'INSERT INTO dbo_algorithmforecast VALUES ({}, {}, {}, {}, {})'

                # populate entire table if empty
                # or add new dates based on information in Statistics table
                """Look into this to add SMA"""
                if latest_date.empty or latest_date['forecastdate'][0] <= data['date'][n]:
                    if data['sMomentum'][n-1] >= 0 and data['lMomentum'][n-1] >= 0:
                        forecastClose = data['close'][n-1] + (2.576 * data['stDev'][n-1] / sqrt(sDev))
                    elif data['sMomentum'][n-1] <= 0 and data['lMomentum'][n-1] <= 0:
                        forecastClose = data['close'][n - 1] + (2.576 * data['stDev'][n - 1] / sqrt(sDev))
                    else:
                        forecastClose = data['movAvg'][n-1]
                    predError = 100 * abs(forecastClose - data['close'][n])/data['close'][n]
                    forecastDate = "'" + str(data['date'][n]) + "'"

                    #insert new prediction into table
                    insert_query = insert_query.format(forecastDate, ID, forecastClose, algoCode, predError)
                    self.engine.execute(insert_query)

            # model for future price movements
            data['momentumA'] = data['close'].diff(10)
            data['lagMomentum'] = data['momentumA'].shift(5)

            fdate = "'" + str(data['date'][n]) + "'"
            # number of weekdays
            weekdays = 10
            # 3 weeks of weekdays
            days = 15
            forecast = []

            forecast_dates_query = 'SELECT date from dbo_datedim WHERE date > {} AND weekend=0 AND isholiday=0 ' \
                                   'ORDER BY date ASC LIMIT {}'.format(fdate, weekdays)
            future_dates = pd.read_sql_query(forecast_dates_query, self.engine)

            insert_query = 'INSERT INTO dbo_algorithmforecast VALUES ({}, {}, {}, {}, {})'

            # Forecast close price tomorrow
            if data['sMomentum'][n] >= 0 and data['lMomentum'][n] >= 0:
                forecastClose = data['close'][n] + (2.576 * data['stDev'][n] / sqrt(sDev))
            elif data['sMomentum'][n] <= 0 and data['lMomentum'][n] <= 0:
                forecastClose = data['close'][n] + (2.576 * data['stDev'][n] / sqrt(sDev))
            else:
                forecastClose = data['movAvg'][n]
            predError = 0
            forecastDate = "'" + str(future_dates['date'][0]) + "'"
            insert_query = insert_query.format(forecastDate, ID, forecastClose, algoCode, predError)
            self.engine.execute(insert_query)

            # forecast next 9 days
            # for i in range # of weekdays
            """Forecast for future from here down"""
            for i in range(1, len(future_dates)):

                insert_query = 'INSERT INTO dbo_algorithmforecast VALUES ({}, {}, {}, {}, {})'

                # if the momentum is negative
                if data['momentumA'].tail(1).iloc[0] < 0.00:

                    # Set Fibonacci extensions accordingly
                    data['fibExtHighNeg'] = data['lpeak'] - (
                            (data['lpeak'] - data['ltrough']) * 1.236)
                    data['fibExtLowNeg'] = data['lpeak'] - (
                            (data['lpeak'] - data['ltrough']) * 1.382)
                    highfrllinelong = data['highfrllinelong'].tail(1).iloc[0]

                    # Compute average over last 3 weeks of weekdays
                    avg_days = np.average(data['close'].tail(days))

                    # Compute standard Deviation over the last 3 weeks and the average.
                    std_days = stdev(data['close'].tail(days), avg_days)

                    # Compute Standard Error and apply to variable decrease
                    # assign CMA and EMA values
                    decrease = avg_days - (1.960 * std_days) / (sqrt(days))
                    data['fibExtHighPos'] = 0
                    data['fibExtLowPos'] = 0
                    l_cma = data['lcma'].tail(1)
                    l_cma = l_cma.values[0]
                    l_ema = data['lema'].tail(1)
                    l_ema = l_ema.values[0]

                    # Loop through each upcoming day in the week
                    for x in range(weekdays-1):

                        # Compare to current location of cma and frl values
                        # if CMA and FRL are lower than forecast
                        # Forecast lower with a medium magnitude
                        if decrease > l_cma or decrease >= (highfrllinelong + (highfrllinelong * 0.01)) \
                                or decrease > l_ema:
                            decrease -= .5 * std_days
                            forecast.append(decrease)

                        # If CMA and FRL are higher than forecast
                        # Forecast to rise with an aggressive magnitude
                        elif decrease <= l_cma and decrease <= (
                                highfrllinelong - (highfrllinelong * 0.01)) and decrease <= l_ema:
                            decrease += 1.5 * std_days
                            forecast.append(decrease)

                    x = x + 1

                # if the momentum is positive
                elif data['momentumA'].tail(1).iloc[0] > 0.00:
                    # ...Set fibonacci extensions accordingly
                    data['fibExtHighPos'] = data['lpeak'] + (
                            (data['lpeak'] - data['ltrough']) * 1.236)
                    data['fibExtLowPos'] = data['lpeak'] + (
                            (data['lpeak'] - data['ltrough']) * 1.382)
                    highfrllinelong = data['highfrllinelong'].tail(1).iloc[0]

                    # Compute average over last 3 weeks of weekdays
                    avg_days = np.average(data['close'].tail(days))

                    # Compute standard Deviation over the last 3 weeks and the average.
                    std_days = stdev(data['close'].tail(days), avg_days)

                    # Compute Standard Error and apply to variable increase
                    increase = avg_days + (1.960 * std_days) / (sqrt(days))
                    data['fibExtHighNeg'] = 0
                    data['fibExtLowNeg'] = 0
                    l_cma = data['lcma'].tail(1)
                    l_cma = l_cma.values[0]
                    l_ema = data['lema'].tail(1)
                    l_ema = l_ema.values[0]

                    for x in range(weekdays-1):

                        # Compare to current location of cma and frl values
                        # if CMA and FRL are lower than forecast
                        # Forecast lower with a normal magnitude
                        if increase > l_cma and increase >= (highfrllinelong - (highfrllinelong * 0.01)) \
                                and increase > l_ema:
                            increase -= std_days
                            forecast.append(increase)

                        # if CMA and FRL are lower than forecast
                        # Forecast lower with an aggressive magnitude
                        elif increase <= l_cma or increase <= (
                                highfrllinelong - (highfrllinelong * 0.01)) or increase <= l_ema:
                            increase += 1.5 * std_days
                            forecast.append(increase)

                forecastDateStr = "'" + str(future_dates['date'][i]) + "'"
                # Send the addition of new variables to SQL

                # predicted values error is 0 because the actual close prices for the future is not available
                predError = 0

                insert_query = insert_query.format(forecastDateStr, ID, forecast[i], algoCode, predError)
                self.engine.execute(insert_query)

    """Look into why warnings due to incorrect inputs"""
    def calculate_arima_forecast(self):
        """
        Calculate historic next-day returns based on ARIMA forecast model
        and 10 days of future price forecast
        Store results in dbo_AlgorithmForecast
        To predict next day's value, prior 50 business day's close prices are used
        """

        # retrieve InstrumentsMaster table from database
        query = 'SELECT * FROM {}'.format(self.table_name)
        df = pd.read_sql_query(query, self.engine)
        algoCode = "'ARIMA'"

        # add code to database if it doesn't exist
        code_query = 'SELECT COUNT(*) FROM dbo_algorithmmaster WHERE algorithmcode=%s' % algoCode
        count = pd.read_sql_query(code_query, self.engine)
        if count.iat[0, 0] == 0:
            algoName = "'ARIMA'"
            insert_code_query = 'INSERT INTO dbo_algorithmmaster VALUES({},{})'.format(algoCode, algoName)
            self.engine.execute(insert_code_query)

        # loop through each ticker symbol
        for ID in df['instrumentid']:

            # remove all future prediction dates
            remove_future_query = 'DELETE FROM dbo_algorithmforecast WHERE algorithmcode={} AND prederror=0 AND ' \
                                  'instrumentid={}'.format(algoCode, ID)
            self.engine.execute(remove_future_query)

            # find the latest forecast date
            date_query = 'SELECT forecastdate FROM dbo_algorithmforecast WHERE algorithmcode={} AND instrumentid={} ' \
                         'ORDER BY forecastdate DESC LIMIT 1'.format(algoCode, ID)
            latest_date = pd.read_sql_query(date_query, self.engine)  # most recent forecast date calculation

            # if table has forecast prices already find the latest one and delete it
            # need to use most recent data for today if before market close at 4pm
            if not latest_date.empty:
                latest_date_str = "'" + str(latest_date['forecastdate'][0]) + "'"
                delete_query = 'DELETE FROM dbo_algorithmforecast WHERE algorithmcode={} AND instrumentid={} AND ' \
                               'forecastdate={}'.format(algoCode, ID, latest_date_str)
                self.engine.execute(delete_query)

            # get raw price data from database
            data_query = 'SELECT date, close FROM dbo_instrumentstatistics WHERE instrumentid=%s ORDER BY Date ASC' % ID
            data = pd.read_sql_query(data_query, self.engine)
            """Below here to look at for ARIMA warnings and to tweak"""
            # training data size
            # IF THIS CHANGES ALL PREDICTIONS STORED IN DATABASE BECOME INVALID!
            input_length = 10

            for n in range((input_length-1), len(data)):
                insert_query = 'INSERT INTO dbo_algorithmforecast VALUES ({}, {}, {}, {}, {})'

                # populate entire table if empty
                # or add new dates based on information in Statistics table

                if latest_date.empty or latest_date['forecastdate'][0] <= data['date'][n]:
                    training_data = data['close'][n-(input_length-1):n]
                    arima = ARIMA(training_data, order=(0,1,0))    # most suited order combination after many trials
                    fitted_arima = arima.fit(disp=-1)
                    forecastClose = data['close'][n] + fitted_arima.fittedvalues[n-1]

                    predError = 100 * abs(forecastClose - data['close'][n]) / data['close'][n]
                    forecastDate = "'" + str(data['date'][n]) + "'"

                    insert_query = insert_query.format(forecastDate, ID, forecastClose, algoCode, predError)
                    self.engine.execute(insert_query)

            # training and test data set sizes
            forecast_length = 10
            forecast_input = 50

            # find ARIMA model for future price movements
            training_data = data['close'][-forecast_input:]
            model = ARIMA(training_data, order=(0, 1, 0))
            fitted = model.fit(disp=0)
            fc, se, conf = fitted.forecast(forecast_length, alpha=0.5)

            forecast_dates_query = 'SELECT date from dbo_datedim WHERE date > {} AND weekend=0 AND isholiday=0 ' \
                                   'ORDER BY date ASC LIMIT {}'.format(forecastDate, forecast_length)
            future_dates = pd.read_sql_query(forecast_dates_query, self.engine)

            # insert prediction into database
            date = data['date'][n]
            for n in range(0, forecast_length):
                insert_query = 'INSERT INTO dbo_algorithmforecast VALUES ({}, {}, {}, {}, {})'
                forecastClose = fc[n]
                predError = 0
                forecastDate = "'" + str(future_dates['date'][n]) + "'"
                insert_query = insert_query.format(forecastDate, ID, forecastClose, algoCode, predError)
                self.engine.execute(insert_query)

    def calculate_random_forest_forecast(self):
        """
        Calculate historic next-day returns based on Random Forest forecast model
        and 10 days of future price forecast
        Store results in dbo_AlgorithmForecast table in the database
        """

        # retrieve InstrumentsMaster table from database
        query = 'SELECT * FROM {}'.format(self.table_name)
        df = pd.read_sql_query(query, self.engine)
        algoCode = "'RandomForest'"

        # add code to database if it doesn't exist
        code_query = 'SELECT COUNT(*) FROM dbo_algorithmmaster WHERE algorithmcode=%s' % algoCode
        count = pd.read_sql_query(code_query, self.engine)
        if count.iat[0, 0] == 0:
            algoName = "'RandomForest'"
            insert_code_query = 'INSERT INTO dbo_algorithmmaster VALUES({},{})'.format(algoCode, algoName)
            self.engine.execute(insert_code_query)

        # loop through each ticker symbol
        for ID in df['instrumentid']:
            # remove all future prediction dates - these need to be recalculated daily
            remove_future_query = 'DELETE FROM dbo_algorithmforecast WHERE algorithmcode={} AND prederror=0 AND ' \
                                  'instrumentid={}'.format(algoCode, ID)
            self.engine.execute(remove_future_query)

            # find the latest forecast date
            date_query = 'SELECT forecastdate FROM dbo_algorithmforecast WHERE algorithmcode={} AND instrumentid={} ' \
                         'ORDER BY forecastdate DESC LIMIT 1'.format(algoCode, ID)
            latest_date = pd.read_sql_query(date_query, self.engine)  # most recent forecast date calculation

            # if table has forecast prices already find the latest one and delete it
            # need to use most recent data for today if before market close at 4pm
            if not latest_date.empty:
                latest_date_str = "'" + str(latest_date['forecastdate'][0]) + "'"
                delete_query = 'DELETE FROM dbo_algorithmforecast WHERE algorithmcode={} AND instrumentid={} AND ' \
                               'forecastdate={}'.format(algoCode, ID, latest_date_str)
                self.engine.execute(delete_query)

            # get raw price data from database
            data_query = 'SELECT date, close FROM dbo_instrumentstatistics WHERE instrumentid=%s ORDER BY Date ASC' % ID
            data = pd.read_sql_query(data_query, self.engine)

            # training data size
            # IF THIS CHANGES ALL PREDICTIONS STORED IN DATABASE BECOME INVALID!
            input_length = 10

            for n in range((input_length - 1), len(data)):
                insert_query = 'INSERT INTO dbo_algorithmforecast VALUES ({}, {}, {}, {}, {})'

                # populate entire table if empty
                # or add new dates based on information in Statistics table
                if latest_date.empty or latest_date['forecastdate'][0] <= data['date'][n]:

                    # historical next-day random forest forecast
                    x_train = [i for i in range(input_length-1)]
                    y_train = data['close'][n - (input_length - 1):n]
                    x_test = [input_length-1]

                    x_train = np.array(x_train)
                    y_train = np.array(y_train)
                    x_test = np.array(x_test)
                    x_train = x_train.reshape(-1, 1)
                    x_test = x_test.reshape(-1, 1)

                    clf_rf = RandomForestRegressor(n_estimators=100)   # meta estimator with classifying decision trees
                    clf_rf.fit(x_train, y_train)                       # x and y train fit into classifier
                    forecastClose = clf_rf.predict(x_test)[0]
                    predError = 100 * abs(forecastClose-data['close'][n])/data['close'][n]   # standard MBE formula
                    forecastDate = "'" + str(data['date'][n]) + "'"

                    insert_query = insert_query.format(forecastDate, ID, forecastClose, algoCode, predError)
                    self.engine.execute(insert_query)

            # training and test data set sizes
            forecast_length = 10
            forecast_input = 50

            # find Random Forest model for future price movements
            x_train = [i for i in range(forecast_input)]
            y_train = data['close'][-forecast_input:]
            x_test = [i for i in range(forecast_length)]

            x_train = np.array(x_train)
            y_train = np.array(y_train)
            x_test = np.array(x_test)
            x_train = x_train.reshape(-1, 1)
            x_test = x_test.reshape(-1, 1)

            clf_rf = RandomForestRegressor(n_estimators=100)
            clf_rf.fit(x_train, y_train)
            forecast = clf_rf.predict(x_test)

            forecast_dates_query = 'SELECT date from dbo_datedim WHERE date > {} AND weekend=0 AND isholiday=0 ' \
                                   'ORDER BY date ASC LIMIT {}'.format(forecastDate, forecast_length)
            future_dates = pd.read_sql_query(forecast_dates_query, self.engine)

            # insert prediction into database
            date = data['date'][n]
            for n in range(0, forecast_length):
                insert_query = 'INSERT INTO dbo_algorithmforecast VALUES ({}, {}, {}, {}, {})'
                forecastClose = forecast[n]
                predError = 0
                forecastDate = "'" + str(future_dates['date'][n]) + "'"
                insert_query = insert_query.format(forecastDate, ID, forecastClose, algoCode, predError)
                self.engine.execute(insert_query)

    """Delete Forecast old"""
    def calculate_forecast_old(self):
        """
        Calculate historic one day returns based on traditional forecast model
        and 10 days of future price forecast
        Store results in dbo_AlgorithmForecast
        This method was from Winter 2019 or before and is not really useful because
        it uses each day's actual close price (after the market closes) to predict that day's close price -
        it is only included for comparison with our improved `PricePred` algorithm`
        Prior 10 days close prices are used to predict the price for the next day
        """
        # retrieve InstrumentsMaster table from database
        query = 'SELECT * FROM {}'.format(self.table_name)
        df = pd.read_sql_query(query, self.engine)
        algoCode = "'PricePredOld'"

        # add code to database if it doesn't exist
        code_query = 'SELECT COUNT(*) FROM dbo_algorithmmaster WHERE algorithmcode=%s' % algoCode
        count = pd.read_sql_query(code_query, self.engine)
        if count.iat[0, 0] == 0:
            algoName = "'PricePredictionOld'"
            insert_code_query = 'INSERT INTO dbo_algorithmmaster VALUES({},{})'.format(algoCode, algoName)
            self.engine.execute(insert_code_query)

        # loop through each ticker symbol
        for ID in df['instrumentid']:

            # remove all future prediction dates
            remove_future_query = 'DELETE FROM dbo_algorithmforecast WHERE algorithmcode={} AND prederror=0 AND ' \
                                  'instrumentid={}'.format(algoCode, ID)
            self.engine.execute(remove_future_query)

            # find the latest forecast date
            date_query = 'SELECT forecastdate FROM dbo_algorithmforecast WHERE algorithmcode={} AND instrumentid={} ' \
                         'ORDER BY forecastdate DESC LIMIT 1'.format(algoCode, ID)
            latest_date = pd.read_sql_query(date_query, self.engine)  # most recent forecast date calculation

            # if table has forecast prices already find the latest one and delete it
            # need to use most recent data for today when market closes at 4pm, not before that
            if not latest_date.empty:
                latest_date_str = "'" + str(latest_date['forecastdate'][0]) + "'"
                delete_query = 'DELETE FROM dbo_algorithmforecast WHERE algorithmcode={} AND instrumentid={} AND ' \
                               'forecastdate={}'.format(algoCode, ID, latest_date_str)
                self.engine.execute(delete_query)

            # get raw price data from database
            data_query = 'SELECT date, close FROM dbo_instrumentstatistics WHERE instrumentid=%s ORDER BY Date ASC' % ID
            data = pd.read_sql_query(data_query, self.engine)

            # prediction formula inputs
            # IF THESE CHANGE ALL RELATED PREDICTIONS STORED IN DATABASE BECOME INVALID!
            momentum = 5
            sDev = 10
            ma = 10
            start = max(momentum, sDev, ma)

            # calculate prediction inputs
            data['momentum'] = data['close'].diff(momentum)
            data['stDev'] = data['close'].rolling(sDev).std()
            data['movAvg'] = data['close'].rolling(ma).mean()

            # first predictions can me made after 'start' number of days, its 10 days
            for n in range(start, len(data)):
                insert_query = 'INSERT INTO dbo_algorithmforecast VALUES ({}, {}, {}, {}, {})'

                # populate entire table if empty
                # or add new dates based on information in Statistics table
                if latest_date.empty or latest_date['forecastdate'][0] <= data['date'][n]:
                    if data['momentum'][n] >= 0:
                        forecastClose = data['close'][n] + (2.576 * data['stDev'][n] / sqrt(sDev))
                    else:
                        forecastClose = data['close'][n] - (2.576 * data['stDev'][n] / sqrt(sDev))

                    predError = 100 * abs(forecastClose - data['close'][n]) / data['close'][n]
                    forecastDate = "'" + str(data['date'][n]) + "'"

                    # insert new prediction into table
                    insert_query = insert_query.format(forecastDate, ID, forecastClose, algoCode, predError)
                    self.engine.execute(insert_query)

    """Use these forecast to generate buy sell signals"""
    def calculate_svm_forecast(self):
        """
        Calculate historic next-day returns based on SVM
        and 10 days of future price forecast
        Store results in dbo_AlgorithmForecast
        Each prediction is made using prior 10 business days' close prices
        """
        # retrieve InstrumentsMaster table from database
        query = 'SELECT * FROM {}'.format(self.table_name)
        df = pd.read_sql_query(query, self.engine)
        algoCode = "'svm'"

        # add code to database if it doesn't exist
        code_query = 'SELECT COUNT(*) FROM dbo_algorithmmaster WHERE algorithmcode=%s' % algoCode
        count = pd.read_sql_query(code_query, self.engine)
        if count.iat[0, 0] == 0:
            algoName = "'SVM'"
            insert_code_query = 'INSERT INTO dbo_algorithmmaster VALUES({},{})'.format(algoCode, algoName)
            self.engine.execute(insert_code_query)

        # loop through each ticker symbol
        for ID in df['instrumentid']:
            # remove all future prediction dates - these need to be recalculated daily
            remove_future_query = 'DELETE FROM dbo_AlgorithmForecast WHERE AlgorithmCode={} AND PredError=0 AND ' \
                                  'InstrumentID={}'.format(algoCode, ID)
            self.engine.execute(remove_future_query)

            # find the latest forecast date
            date_query = 'SELECT ForecastDate FROM dbo_AlgorithmForecast WHERE AlgorithmCode={} AND InstrumentID={} ' \
                         'ORDER BY ForecastDate DESC LIMIT 1'.format(algoCode, ID)
            latest_date = pd.read_sql_query(date_query, self.engine)  # most recent forecast date calculation

            # if table has forecast prices already find the latest one and delete it
            # need to use most recent data for today if before market close at 4pm
            if not latest_date.empty:
                latest_date_str = "'" + str(latest_date['ForecastDate'][0]) + "'"
                delete_query = 'DELETE FROM dbo_AlgorithmForecast WHERE AlgorithmCode={} AND InstrumentID={} AND ' \
                               'ForecastDate={}'.format(algoCode, ID, latest_date_str)
                self.engine.execute(delete_query)

            # get raw price data from database
            data_query = 'SELECT Date, Close FROM dbo_InstrumentStatistics WHERE InstrumentID=%s ORDER BY Date ASC' % ID
            data = pd.read_sql_query(data_query, self.engine)

            # training data size
            # IF THIS CHANGES ALL PREDICTIONS STORED IN DATABASE BECOME INVALID!
            input_length = 10

            for n in range((input_length - 1), len(data)):
                insert_query = 'INSERT INTO dbo_AlgorithmForecast VALUES ({}, {}, {}, {}, {})'

                # populate entire table if empty
                # or add new dates based on information in Statistics table
                if latest_date.empty or latest_date['ForecastDate'][0] <= data['Date'][n]:
                    # historical next-day random forest forecast
                    x_train = [i for i in range(input_length-1)]
                    y_train = data['Close'][n - (input_length - 1):n]
                    x_test = [input_length-1]

                    x_train = np.array(x_train)
                    y_train = np.array(y_train)
                    x_test = np.array(x_test)
                    x_train = x_train.reshape(-1, 1)
                    x_test = x_test.reshape(-1, 1)

                    clf_svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
                    clf_svr.fit(x_train, y_train)
                    forecastClose = clf_svr.predict(x_test)[0]
                    predError = 100 * abs(forecastClose-data['Close'][n])/data['Close'][n]
                    forecastDate = "'" + str(data['Date'][n]) + "'"

                    insert_query = insert_query.format(forecastDate, ID, forecastClose, algoCode, predError)
                    self.engine.execute(insert_query)

            # training and test data set sizes
            forecast_length = 10
            forecast_input = 50

            # Train Random Forest model for future price movements
            x_train = [i for i in range(forecast_input)]
            y_train = data['Close'][-forecast_input:]
            x_test = [i for i in range(forecast_length)]

            x_train = np.array(x_train)
            y_train = np.array(y_train)
            x_test = np.array(x_test)
            x_train = x_train.reshape(-1, 1)
            x_test = x_test.reshape(-1, 1)

            clf_svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
            clf_svr.fit(x_train, y_train)
            forecast = clf_svr.predict(x_test)

            forecast_dates_query = 'SELECT date from dbo_datedim WHERE date > {} AND weekend=0 AND isholiday=0 ' \
                                   'ORDER BY date ASC LIMIT {}'.format(forecastDate, forecast_length)
            future_dates = pd.read_sql_query(forecast_dates_query, self.engine)

            # insert prediction into database
            for n in range(0, forecast_length):
                insert_query = 'INSERT INTO dbo_algorithmforecast VALUES ({}, {}, {}, {}, {})'
                forecastClose = forecast[n]
                predError = 0
                forecastDate = "'" + str(future_dates['date'][n]) + "'"
                insert_query = insert_query.format(forecastDate, ID, forecastClose, algoCode, predError)
                self.engine.execute(insert_query)

    def calculate_xgboost_forecast(self):
        """
        Calculate historic next-day returns based on XGBoost
        and 10 days of future price forecast
        Store results in dbo_AlgorithmForecast
        Each prediction is made using the prior 50 days close prices
        """
        # retrieve InstrumentsMaster table from database
        query = 'SELECT * FROM {}'.format(self.table_name)
        df = pd.read_sql_query(query, self.engine)
        algoCode = "'xgb'"

        # add code to database if it doesn't exist
        code_query = 'SELECT COUNT(*) FROM dbo_algorithmmaster WHERE algorithmcode=%s' % algoCode
        count = pd.read_sql_query(code_query, self.engine)
        if count.iat[0, 0] == 0:
            algoName = "'xgb'"
            insert_code_query = 'INSERT INTO dbo_algorithmmaster VALUES({},{})'.format(algoCode, algoName)
            self.engine.execute(insert_code_query)

        # loop through each ticker symbol
        for ID in df['instrumentid']:
            # remove all future prediction dates - these need to be recalculated daily
            remove_future_query = 'DELETE FROM dbo_AlgorithmForecast WHERE AlgorithmCode={} AND PredError=0 AND ' \
                                  'InstrumentID={}'.format(algoCode, ID)
            self.engine.execute(remove_future_query)

            # find the latest forecast date
            date_query = 'SELECT ForecastDate FROM dbo_AlgorithmForecast WHERE AlgorithmCode={} AND InstrumentID={} ' \
                         'ORDER BY ForecastDate DESC LIMIT 1'.format(algoCode, ID)
            latest_date = pd.read_sql_query(date_query, self.engine)  # most recent forecast date calculation

            # if table has forecast prices already find the latest one and delete it
            # need to use most recent data for today if before market close at 4pm
            if not latest_date.empty:
                latest_date_str = "'" + str(latest_date['ForecastDate'][0]) + "'"
                delete_query = 'DELETE FROM dbo_AlgorithmForecast WHERE AlgorithmCode={} AND InstrumentID={} AND ' \
                               'ForecastDate={}'.format(algoCode, ID, latest_date_str)
                self.engine.execute(delete_query)

            # get raw price data from database
            data_query = 'SELECT Date, Close FROM dbo_InstrumentStatistics WHERE InstrumentID=%s ORDER BY Date ASC' % ID
            data = pd.read_sql_query(data_query, self.engine)

            # training data size
            # IF THIS CHANGES ALL RELATED PREDICTIONS STORED IN THE DATABASE BECOME INVALID!
            input_length = 10

            for n in range((input_length - 1), len(data)):
                insert_query = 'INSERT INTO dbo_AlgorithmForecast VALUES ({}, {}, {}, {}, {})'
                # populate entire table if empty
                # or add new dates based on information in Statistics table
                if latest_date.empty or latest_date['ForecastDate'][0] <= data['Date'][n]:
                    # historical next-day random forest forecast
                    x_train = [i for i in range(input_length-1)]
                    y_train = data['Close'][n - (input_length - 1):n]
                    x_test = [input_length-1]

                    x_train = np.array(x_train)
                    y_train = np.array(y_train)
                    x_test = np.array(x_test)
                    x_train = x_train.reshape(-1, 1)
                    x_test = x_test.reshape(-1, 1)

                    #XG BOOST Regressor with tree depth, subsample ratio of tree growth...etc.
                    xg_reg = xgb.XGBRegressor(max_depth=3, learning_rate=0.30, n_estimators=15,
                                              objective="reg:linear", subsample=0.5,
                                              colsample_bytree=0.8, seed=10)
                    xg_reg.fit(x_train, y_train)

                    forecastClose = xg_reg.predict(x_test)[0]
                    predError = 100 * abs(forecastClose-data['Close'][n])/data['Close'][n]
                    forecastDate = "'" + str(data['Date'][n]) + "'"

                    insert_query = insert_query.format(forecastDate, ID, forecastClose, algoCode, predError)
                    self.engine.execute(insert_query)

            # training and test data set sizes
            forecast_length = 10
            forecast_input = 50

            # find XG BOOST model for future price movements
            x_train = [i for i in range(forecast_input)]
            y_train = data['Close'][-forecast_input:]
            x_test = [i for i in range(forecast_length)]

            x_train = np.array(x_train)
            y_train = np.array(y_train)
            x_test = np.array(x_test)
            x_train = x_train.reshape(-1, 1)
            x_test = x_test.reshape(-1, 1)

            #XGBoost Regressor Predictions added 11/16/19
            xg_reg = xgb.XGBRegressor(max_depth=3, learning_rate=0.30, n_estimators=15,
                                      objective="reg:linear", subsample=0.5,
                                      colsample_bytree=0.8, seed=10)

            xg_reg.fit(x_train, y_train)
            forecast = xg_reg.predict(x_test)

            forecast_dates_query = 'SELECT date from dbo_datedim WHERE date > {} AND weekend=0 AND isholiday=0 ' \
                                   'ORDER BY date ASC LIMIT {}'.format(forecastDate, forecast_length)
            future_dates = pd.read_sql_query(forecast_dates_query, self.engine)

            # insert prediction into MySQL database
            # predError will be 0, there are no close prices available for future dates
            for n in range(0, forecast_length):
                insert_query = 'INSERT INTO dbo_algorithmforecast VALUES ({}, {}, {}, {}, {})'
                forecastClose = forecast[n]
                predError = 0
                forecastDate = "'" + str(future_dates['date'][n]) + "'"
                insert_query = insert_query.format(forecastDate, ID, forecastClose, algoCode, predError)
                self.engine.execute(insert_query)

    def calculate_regression(self):
        """
            Calculate polynomial regression of the next 10 days
            Algorithm's accuracy is... questionable
        """
        # retrieve InstrumentsMaster table from database
        query = 'SELECT * FROM {}'.format(self.table_name)
        df = pd.read_sql_query(query, self.engine)
        algoCode = "'regression'"

        # add code to database if it doesn't exist
        code_query = 'SELECT COUNT(*) FROM dbo_algorithmmaster WHERE algorithmcode=%s' % algoCode
        count = pd.read_sql_query(code_query, self.engine)
        if count.iat[0, 0] == 0:
            algoName = "'PolynomialRegression'"
            insert_code_query = 'INSERT INTO dbo_algorithmmaster VALUES({},{})'.format(algoCode, algoName)
            self.engine.execute(insert_code_query)

        # loop through each ticker symbol
        for ID in df['instrumentid']:

            # remove all future prediction dates
            remove_future_query = 'DELETE FROM dbo_algorithmforecast WHERE algorithmcode={} AND prederror=0 AND ' \
                                  'instrumentid={}'.format(algoCode, ID)
            self.engine.execute(remove_future_query)

            # find the latest forecast date
            date_query = 'SELECT forecastdate FROM dbo_algorithmforecast WHERE algorithmcode={} AND instrumentid={} ' \
                         'ORDER BY forecastdate DESC LIMIT 1'.format(algoCode, ID)
            latest_date = pd.read_sql_query(date_query, self.engine)  # most recent forecast date calculation

            # if table has forecast prices already find the latest one and delete it
            # need to use most recent data for today if before market close at 4pm
            if not latest_date.empty:
                latest_date_str = "'" + str(latest_date['forecastdate'][0]) + "'"
                delete_query = 'DELETE FROM dbo_algorithmforecast WHERE algorithmcode={} AND instrumentid={} AND ' \
                               'forecastdate={}'.format(algoCode, ID, latest_date_str)
                self.engine.execute(delete_query)

            # get raw price data from database
            data_query = 'SELECT date, close FROM dbo_instrumentstatistics WHERE instrumentid=%s ORDER BY Date ASC' % ID
            data = pd.read_sql_query(data_query, self.engine)

            # regression model from previous days
            input_length = 20

            # predict ahead
            forecast_length = 5

            for n in range(input_length, len(data)):

                recent_data = data[n - input_length:n]

                # get most recent trading day
                forecastDate = "'" + str(data['date'][n]) + "'"

                # x and y axis
                x_axis = np.array(recent_data['date'])
                y_axis = np.array(recent_data['close'])

                # convert date to a ordinal value to allow for regression
                df = pd.DataFrame({'date': x_axis, 'close': y_axis})
                df['date'] = pd.to_datetime(df['date'])
                df['date'] = df['date'].map(dt.datetime.toordinal)

                X = np.array(df['date'])
                X = np.array(X)
                X = X.reshape(-1, 1)
                y = np.array(df['close'])

                poly_reg = PolynomialFeatures(degree=4)
                X_poly = poly_reg.fit_transform(X)
                pol_reg = LinearRegression()
                pol_reg.fit(X_poly, y)
                # plt.scatter(X, y, color='red')
                # plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='blue')
                # plt.title('Prediction')
                # plt.xlabel('Date')
                # plt.ylabel('Percentage Change')
                # plt.show()

                forecast_dates_query = 'SELECT date from dbo_datedim WHERE date > {} AND weekend=0 AND isholiday=0 ' \
                                       'ORDER BY date ASC LIMIT {}'.format(forecastDate, forecast_length)

                future_dates = pd.read_sql_query(forecast_dates_query, self.engine)
                # delete outdated forecasts for the next period
                delete_query = 'DELETE FROM dbo_algorithmforecast WHERE algorithmcode={} AND instrumentid={} AND ' \
                               'forecastdate>{}'.format(algoCode, ID, forecastDate)
                self.engine.execute(delete_query)

                for n in range(len(future_dates)):
                    insert_query = 'INSERT INTO dbo_algorithmforecast VALUES ({}, {}, {}, {}, {})'

                    forecastDate = future_dates['date'][n]

                    ordinalDate = forecastDate.toordinal()
                    forecastDate = "'" + str(future_dates['date'][n]) + "'"

                    forecastClose = pol_reg.predict(poly_reg.fit_transform([[ordinalDate]]))
                    forecastClose = (round(forecastClose[0], 3))
                    # populate entire table if empty
                    # or add new dates based on information in Statistics table
                    insert_query = insert_query.format(forecastDate, ID, forecastClose, algoCode, 0)

                    self.engine.execute(insert_query)
    def MSF1(self):
        #Queires the database to grab all of the Macro Economic Variable codes
        query = "SELECT macroeconcode FROM dbo_macroeconmaster WHERE activecode = 'A'"
        id = pd.read_sql_query(query, self.engine)
        id = id.reset_index(drop=True)

        #Queries the database to grab all of the instrument IDs
        query = 'SELECT instrumentid FROM dbo_instrumentmaster'
        id2 = pd.read_sql_query(query, self.engine)
        id2 = id2.reset_index(drop = True)

        # Sets value for number of datapoints you would like to work with
        #Assuming this is the number of predictions the for loop makes
        #Changing this according to changed monthly calculations
        n = 8

        # Getting Dates for Future Forecast#
        #Initialize the currentDate variable for use when grabbing the forecasted dates
        currentDate = datetime.today()

        # Creates a list to store future forecast dates
        date = []

        # This will set the value of count according to which month we are in, this is to avoid having past forecast dates in the list
        #Changed to month wise calculations
        if (currentDate.month < 4):
            count = 0
        elif (currentDate.month < 7 and currentDate.month >= 4):
            count = 1
        elif (currentDate.month < 10 and currentDate.month >= 7):
            count = 2
        else:
            count = 3

        # Initialize a variable to the current year
        year = currentDate.year

        #Prints out the accuracy figures, not necessary can be commented out
        FinsterTab.W2020.AccuracyTest.MSF1_accuracy(self.engine)

        # Setup a for loop to loop through and append the date list with the date of the start of the next quarter
        # For loop will run n times, corresponding to amount of data points we are working with
        for i in range(n):

            # If the count is 0 then we are still in the first quarter
            if (count == 0):
                # Append the date list with corresponding quarter and year
                date.append(str(year) + "-03-" + "31")
                # Increase count so this date is not repeated for this year
                count += 1

            #Do it again for the next quarter
            elif (count == 1):
                date.append(str(year) + "-06-" + "30")
                count += 1

            #And again for the next quarter
            elif (count == 2):
                date.append(str(year) + "-09-" + "30")
                count += 1

            # Until we account for the last quarter of the year
            else:
                date.append(str(year) + "-12-" + "31")
                count = 0
                # Where we then incrament the year for the next iterations
                year = year + 1

        #Initializes a list for which we will eventually be storing all data to add to the macroeconalgorithm database table
        data = []

        #Create a for loop to iterate through all of the instrument ids
        for v in id2['instrumentid']:

            #Median_forecast will be a dictionary where the key is the date and the value is a list of forecasted prices
            median_forecast = {}
            #This will be used to easily combine all of the forecasts for different dates to determine the median forecast value
            for i in date:
                temp = {i: []}
                median_forecast.update(temp)

            # Initiailizes a variable to represent today's date, used to fetch forecast dates
            currentDate = str(datetime.today())
            # Applies quotes to current date so it can be read as a string
            currentDate = ("'" + currentDate + "'")


            #This query will grab quarterly instrument prices from between 2014 and the current date to be used in the forecasting
            query = "SELECT close, instrumentid FROM ( SELECT date, close, instrumentID, ROW_NUMBER() OVER " \
                    "(PARTITION BY YEAR(date), MONTH(date) ORDER BY DAY(date) DESC) AS rowNum FROM " \
                    "dbo_instrumentstatistics WHERE instrumentid = {} AND date BETWEEN '2014-03-21' AND {} ) z " \
                    "WHERE rowNum = 1 AND ( MONTH(z.date) = 3 OR MONTH(z.date) = 6 OR MONTH(z.date) = 9 OR " \
                    "MONTH(z.date) = 12)".format(v, currentDate)

            # Executes the query and stores the result in a dataframe variable
            df2 = pd.read_sql_query(query, self.engine)

            #This for loop iterates through the different macro economic codes to calculate the percent change for each macroeconomic variable
            for x in id['macroeconcode']:

                #Retrieves Relevant Data from Database

                query = 'SELECT * FROM dbo_macroeconstatistics WHERE macroeconcode = {}'.format('"' + str(x) + '"')
                df = pd.read_sql_query(query, self.engine)
                macro = df.tail(n)
                SP = df2.tail(n)
                temp = df.tail(n+1)
                temp = temp.reset_index()

                #Converts macro variables to precent change
                macroPercentChange = macro
                macro = macro.reset_index(drop=True)
                SP = SP.reset_index(drop=True)
                macroPercentChange = macroPercentChange.reset_index(drop=True)


                for i in range(0, n):

                    if (i == 0):
                        macrov = (macro['statistics'][i]-temp['statistics'][i])/temp['statistics'][i]
                        macroPercentChange['statistics'].iloc[i] = macrov * 100
                    else:
                        macrov = (macro['statistics'][i]-macro['statistics'][i - 1])/macro['statistics'][i - 1]
                        macroPercentChange['statistics'].iloc[i] = macrov * 100


                #Algorithm for forecast price
                S = DataForecast.calc(self, macroPercentChange, SP, n) #Calculates the average GDP and S&P values for the given data points over n days and performs operations on GDP average

                # temp_price will be used to hold the previous forecast price for the next prediction
                temp_price = 0

                # isFirst will determine whether or not this is the first calculation being done
                # If it is true then we use the most recent instrument statistic to forecast the first pricepoint
                # IF it is false then we use the previous forecast price to predict the next forecast price
                isFirst = True

                # Setup a for loop to calculate the final forecast price and add data to the list variable data
                for i in range(n):
                    if isFirst:
                        if x in [2, 3, 4]:
                            temp_price = ((S*SP['close'].iloc[n-1]) + SP['close'].iloc[n-1])
                            isFirst = False
                        else:
                            temp_price = ((S*SP['close'].iloc[n-1]) + SP['close'].iloc[n-1])
                            isFirst = False
                    else:
                        if x in [2, 3, 4]:
                            temp_price = ((S*temp_price) + (temp_price % n))
                        else:
                            temp_price = ((S*temp_price)+ (temp_price % n))

                    #Once the forecast price is calculated append it to median_forecast list
                    median_forecast[date[i]].append(temp_price)

            #Calculates the median value for each date using a list of prices forecasted by each individual macro economic variable
            forecast_prices = []
            for i in date:
                #Sort the forecasted prices based on date
                sorted_prices = sorted(median_forecast[i])
                #calculate the median forecasted price for each date
                if len(sorted_prices) % 2 == 0:
                    center = int(len(sorted_prices)/2)
                    forecast_prices.append(sorted_prices[center])
                else:
                    center = int(len(sorted_prices)/2)
                    forecast_prices.append((sorted_prices[center] + sorted_prices[center - 1])/2)

            #Set up a for loop to construct a list using variables associated with macroeconalgorithm database table
            for i in range(len(forecast_prices)):
                data.append([date[i], v, 'ALL', forecast_prices[i], 'MSF_new', 0])

        # Convert data list to dataframe variable
        table = pd.DataFrame(data, columns=['forecastdate','instrumentid' , 'macroeconcode',
                                            'forecastprice', 'algorithmcode', 'prederror'])

        #Fill the database with the relevant information
        table.to_sql('dbo_macroeconalgorithmforecast', self.engine, if_exists=('replace'),
                     index=False)

    def MSF_final(self):
        # This algorithm uses weightings strategy for the macro variables so, setWeightings needs to be True
        # The weights function is written separately below this algorithm
        #Accuracy and error functions are separately written below this algorithm

        setWeightings = False
        #Set to false now becuase of the weightings function not applying properly to all the instruments.
        # Query to grab the macroeconcodes and macroeconnames from the macroeconmaster database table
        query = "SELECT macroeconcode, macroeconname FROM dbo_macroeconmaster WHERE activecode = 'A'"
        data = pd.read_sql_query(query, self.engine)
        #To see if the macro data is being pulled properly, uncomment the following line:
        #print(data)

        # Query to grab the instrumentid and instrument name from the instrumentmaster database table
        #We currently have 10 instruments, uncomment the following lines of code to print the list of instruments by order, namely:
        print('General Motors: GM')
        print('Pfizer: PFE')
        print('S&P 500: SPY')
        print('XPH')
        print('CARZ')
        print('^TYX')
        print('Fiat Chrysler Automobiles: FCAU')
        print('Toyota Motors: TM')
        print('Ford: F')
        print('Honda Motor Company: HMC')

        query = 'SELECT instrumentid, instrumentname FROM dbo_instrumentmaster'
        data1 = pd.read_sql_query(query, self.engine)
        #To see of the data is being pulled properly, uncomment the following line:
        #print(data1)

        # Keys is a dictionary that will be used to store the macro econ code for each macro econ name
        keys = {}
        for i in range(len(data)):
            keys.update({data['macroeconname'].iloc[i]: data['macroeconcode'].iloc[i]})

        # ikeys is a dictionary that will be used to store instrument ids for each instrument name
        ikeys = {}
        for x in range(len(data1)):
            ikeys.update({data1['instrumentname'].iloc[x]: data1['instrumentid'].iloc[x]})

        # Vars is a dictionary used to store the macro economic variable percent change for each macro economic code
        vars = {}
        for i in data['macroeconname']:
            # Vars is only populated with the relevant macro economic variables (GDP, UR, IR, MI and COVI)
            if(i == 'GDP' or i == 'Unemployment Rate' or i == 'Inflation Rate' or i == 'Misery Index' or i == 'Crude Oil ETF Volatility Index'):
                d = {i: []}
                vars.update(d)

        # Result will hold the resulting forecast prices for each instrument ID
        result = {}
        for i in data1['instrumentid']:
            d = {i: []}
            result.update(d)


        # Weightings are determined through a function written in accuracytest.py
        # The weightings returned are used in the calculation below
        weightings = FinsterTab.W2020.DataForecast.create_weightings_MSF_final(self.engine, setWeightings)

        n = 12

        # Getting Dates for Future Forecast #
        # --------------------------------------------------------------------------------------------------------------#
        # Initialize the currentDate variable for use when grabbing the forecasted dates
        currentDate = datetime.today()

        # Creates a list to store future forecast dates
        date = []

        # This will set the value of count according to which month we are in, this is to avoid having past forecast dates in the list
        if (currentDate.month < 4):
            count = 0
        elif (currentDate.month < 7 and currentDate.month >= 4):
            count = 1
        elif (currentDate.month < 10 and currentDate.month >= 7):
            count = 2
        else:
            count = 3

        # Initialize a variable to the current year
        year = currentDate.year

        # Setup a for loop to loop through and append the date list with the date of the start of the next quarter
        # For loop will run n times, corresponding to amount of data points we are working with
        for i in range(n):
            # If the count is 0 then we are still in the first quarter
            if (count == 0):
                # Append the date list with corresponding quarter and year
                date.append(str(year) + "-03-" + "31")
                # Increase count so this date is not repeated for this year
                count += 1

            # Do the same for the next quarter
            elif (count == 1):
                date.append(str(year) + "-06-" + "30")
                count += 1

            # And for the next quarter
            elif (count == 2):
                date.append(str(year) + "-09-" + "30")
                count += 1

            # Until we account for the last quarter of the year
            else:
                date.append(str(year) + "-12-" + "31")
                # Where we then reinitialize count to 0
                count = 0
                # And then increment the year for the next iterations
                year = year + 1
        # --------------------------------------------------------------------------------------------------------------#

        # reinitializes currentDate to todays date, also typecasts it to a string so it can be read by MySQL
        currentDate = str(datetime.today())
        currentDate = ("'" + currentDate + "'")

        # For loop to loop through the macroeconomic codes to calculate the macro economic variable percent change
        for i in keys:
            # Check to make sure the macroeconcode we are working with is one of the relevant ones
            if i in vars:
                # Query to grab the macroeconomic statistics from the database using the relevant macro economic codes
                query = 'SELECT date, statistics, macroeconcode FROM dbo_macroeconstatistics WHERE macroeconcode = {}'.format('"' + keys[i] + '"')
                data = pd.read_sql_query(query, self.engine)

                # For loop to retrieve macro statistics and calculate percent change
                for j in range(n):
                    # This will grab the n+1 statistic to use to calculate the percent change to the n statistic
                    temp = data.tail(n + 1)
                    # This will grab the most recent n statistics from the query, as we are working only with n points
                    data = data.tail(n)

                    # For the first iteration we need to use the n+1th statistic to calculate percent change on the oldest point
                    if j == 0:
                        macrov = (data['statistics'].iloc[j] - temp['statistics'].iloc[0]) / temp['statistics'].iloc[0]
                        vars[i].append(macrov)
                    else:
                        macrov = (data['statistics'].iloc[j] - data['statistics'].iloc[j - 1]) / \
                                 data['statistics'].iloc[j - 1]
                        vars[i].append(macrov)

                #To show the (n+1) statistics data collected by temp variable, uncomment the following line:
                #print(temp)

                #To show the most recent n statistics data grabbed by data variable, uncomment the following line:
                #print(data)
        # We now iterate through the instrument ids
        for x in ikeys:

            # This query will grab the quarterly instrument statistics from 2012 to now
            query = "SELECT date, close, instrumentid FROM ( SELECT date, close, instrumentid, ROW_NUMBER() OVER " \
                    "(PARTITION BY YEAR(date), MONTH(date) ORDER BY DAY(date) DESC) AS rowNum FROM " \
                    "dbo_instrumentstatistics WHERE instrumentid = {} AND date BETWEEN '2012-01-01' AND {} ) z " \
                    "WHERE rowNum = 1 AND ( MONTH(z.date) = 3 OR MONTH(z.date) = 6 OR MONTH(z.date) = 9 OR " \
                    "MONTH(z.date) = 12)".format(ikeys[x], currentDate)

            # Then we execute the query and store the returned values in instrumentStats, and grab the last n stats from the dataframe as we are only using n datapoints
            instrumentStats = pd.read_sql_query(query, self.engine)
            instrumentStats = instrumentStats.tail(n)

            #To verify the pulled information, uncomment the following line:
            #print(instrumentStats)

            # Temp result will then store the resulting forecast prices throughout the calculation of n datapoints
            temp_result = []

            # isFirst will determine whether or not this is the first calculation being done
            # If it is true then we use the most recent instrument statistic to forecast the first pricepoint
            # IF it is false then we use the previous forecast price to predict the next forecast price
            isFirst = True
            # This for loop is where the actual calculation takes place
            for i in range(n):
                if isFirst:
                    stat = vars['GDP'][i] * weightings[ikeys[x]][0] - (vars['Unemployment Rate'][i] * weightings[ikeys[x]][1] + vars['Inflation Rate'][i] * weightings[ikeys[x]][2]) - (vars['Misery Index'][i] * vars['Misery Index'][i])
                    stat = (stat * instrumentStats['close'].iloc[n-1]) + instrumentStats['close'].iloc[n-1]
                    temp_result.append(stat)
                    temp_price = stat
                    isFirst = False
                else:
                    stat = vars['GDP'][i] * weightings[ikeys[x]][0] - (vars['Unemployment Rate'][i] * weightings[ikeys[x]][1] + vars['Inflation Rate'][i] *
                                weightings[ikeys[x]][2]) - (vars['Misery Index'][i] * vars['Misery Index'][i])
                    stat = (stat * temp_price) + temp_price
                    temp_result.append(stat)
                    temp_price = stat

            # We then append the resulting forcasted prices over n quarters to result, a dictionary where each
            # Instrument ID is mapped to n forecast prices
            result[ikeys[x]].append(temp_result)

        #Table will represent a temporary table with the data appended matching the columns of the macroeconalgorithmforecast database table
        table = []
        #This forloop will populate table[] with the correct values according to the database structure
        for i, k in result.items():
            cnt = 0
            for j in k:
                for l in range(n):
                    table.append([date[cnt], i, 'ALL', j[cnt], 'MSF_final', 0])
                    cnt += 1

        #Once table is populated we then push it into the macroeconalgorithmforecast table
        table = pd.DataFrame(table, columns=['forecastdate','instrumentid' , 'macroeconcode',
                                            'forecastprice', 'algorithmcode', 'prederror'])
        table.to_sql('dbo_macroeconalgorithmforecast', self.engine, if_exists=('append'), index=False)

        #After the above loop is completed, the macroeconalgorithmforecast table will be populated with quarterly forecast made by MSF_final algorithm code

    def MSF_final_accuracy(self):
        n = 12
    #Used previous semester's accuracy function to fit the style of how the MSF_final function is written
        # Gets the macro economic variables codes and names to loop through the inidividual macro variables
        query = "SELECT macroeconcode, macroeconname FROM dbo_macroeconmaster WHERE activecode = 'A'"
        data = pd.read_sql_query(query, self.engine)
        macrocodes = []
        indicators = {}
        for i in range(len(data['macroeconcode'])):
            macrocodes.append(data['macroeconcode'].loc[i])
            d = {data['macroeconcode'].loc[i]: []}
            indicators.update(d)

        # Gets the instrument ids to loop through the individual instruments
        query = 'SELECT instrumentid, instrumentname FROM dbo_instrumentmaster'
        data = pd.read_sql_query(query, self.engine)
        instrumentids = []
        for i in data['instrumentid']:
            instrumentids.append(i)

        # These date ranges can be changed if needed, for now, I wanted to work with 5 years of previous data
        # start_date represents the starting date for the forecasts and the end of the training dates
        start_date = "'2019-01-01'"
        # end_date represents the date for which the forecasting ends
        end_date = "'2020-01-10'"
        # train_date represents the date to start collecting the instrument statistics used to forecast prices
        train_date = "'2015-01-01'"

        # Loops through each instrument id to preform error calculations 1 instrument at a time
        for i in instrumentids:

            # Gets the instrument statistics to run through the function
            query = "SELECT date, close, instrumentid FROM ( SELECT date, close, instrumentID, ROW_NUMBER() OVER " \
                    "(PARTITION BY YEAR(date), MONTH(date) ORDER BY DAY(date) DESC) AS rowNum FROM " \
                    "dbo_instrumentstatistics WHERE instrumentid = {} AND date BETWEEN {} AND {} ) z " \
                    "WHERE rowNum = 1 AND ( MONTH(z.date) = 3 OR MONTH(z.date) = 6 OR MONTH(z.date) = 9 OR " \
                    "MONTH(z.date) = 12)".format(i, train_date, start_date)
            train_data = pd.read_sql_query(query, self.engine)

            # Gets the instrument statistics to check against the forecast prices
            query = "SELECT date, close, instrumentid FROM ( SELECT date, close, instrumentID, ROW_NUMBER() OVER " \
                    "(PARTITION BY YEAR(date), MONTH(date) ORDER BY DAY(date) DESC) AS rowNum FROM " \
                    "dbo_instrumentstatistics WHERE instrumentid = {} AND date BETWEEN {} AND {} ) z " \
                    "WHERE rowNum = 1 AND ( MONTH(z.date) = 3 OR MONTH(z.date) = 6 OR MONTH(z.date) = 9 OR " \
                    "MONTH(z.date) = 12)".format(i, start_date, end_date)
            check_data = pd.read_sql_query(query, self.engine)

            # Gets the dates for the future forecast prices so they match the instrument statistics
            dates = []
            for l in check_data['date']:
                dates.append(str(l))

            # Loops through the macro economic variable codes to calculate percent change
            for j in macrocodes:
                # Retrieves macro economic statistics for each macro variables
                query = "SELECT date, statistics, macroeconcode FROM dbo_macroeconstatistics WHERE macroeconcode = {} AND date <= {}".format(
                    '"' + j + '"', start_date)
                data = pd.read_sql_query(query, self.engine)

                # For loop to retrieve macro statistics and calculate percent change
                for k in range(n):
                    temp = data.tail(n + 1)
                    data = data.tail(n)
                    if j == k:
                        macrov = (data['statistics'].iloc[k] - temp['statistics'].iloc[0]) / temp['statistics'].iloc[0]
                        indicators[j].append(macrov)
                    else:
                        macrov = (data['statistics'].iloc[k] - data['statistics'].iloc[k - 1]) / \
                                 data['statistics'].iloc[
                                     k - 1]
                        indicators[j].append(macrov)

            # Preforms the actual calculations and stores them in an array called calculated forecast
            calculated_forecast = []
            for k in range(n):
                stat = indicators['GDP'][k] * 1 - (
                            indicators['UR'][k] * 0 + indicators['IR'][k] * .5 + indicators['COVI'][k] * .5) - (
                               indicators['MI'][k] * indicators['MI'][k])
                stat = (stat * train_data['close'].iloc[n - 1]) + train_data['close'].iloc[n - 1]
                calculated_forecast.append(stat)

            #To see the calculated forecasts array at this point, uncomment the following line:
            #print(calculated_forecast)

            # Creates and inserts the forecast dates, instrument ids, calculated forecast prices, and actual close prices into an array
            results = []
            for k in range(n):
                results.append([dates[k], i, calculated_forecast[k], check_data['close'].loc[k]])

            # Creates a dataframe out of the array created above
            df = pd.DataFrame(results, columns=['forecastdate', 'instrumentid', 'forecastcloseprice', 'close'])
            # print(df)

            count = 0
            # Calculates accuracy
            percent_error = []
            temp_error = 0
            for x in range((len(df) - 1)):  # Check if upward or downward trend
                if (df['close'][x + 1] > df['close'][x] and df['forecastcloseprice'][x + 1] > df['forecastcloseprice'][
                    x]) or \
                        (df['close'][x + 1] < df['close'][x] and df['forecastcloseprice'][x + 1] <
                         df['forecastcloseprice'][x]):
                    count += 1
                temp_error = abs((df['close'][x] - df['forecastcloseprice'][x])) / df['close']

            # Percent Error calculation

            temp_error = (df['close'] - df['forecastcloseprice']) / df['close']
            absolute_percent_error = [abs(ele) for ele in temp_error]
            percent_error.append(absolute_percent_error)

            if df['instrumentid'][i] == 1:
                gm_temp_error = (df['close'] - df['forecastcloseprice']) / df['close']
                gm_absolute_percent_error = [abs(ele) for ele in gm_temp_error]

                # Calculate sum of percent error and find average

                gm_average_percent_error = sum(gm_absolute_percent_error) / 12
                print("Average percent error of MSF_final on GM stock is: ", gm_average_percent_error * 100, "%")

            if df['instrumentid'][i] == 2:
                pfe_temp_error = (df['close'] - df['forecastcloseprice']) / df['close']
                pfe_absolute_percent_error = [abs(ele) for ele in pfe_temp_error]

                # Calculate sum of percent error and find average

                pfe_average_percent_error = sum(pfe_absolute_percent_error) / 12
                print("Average percent error of MSF_final on PFE stock is: ", pfe_average_percent_error * 100, "%")

            if df['instrumentid'][i] == 3:
                spy_temp_error = (df['close'] - df['forecastcloseprice']) / df['close']
                spy_absolute_percent_error = [abs(ele) for ele in spy_temp_error]

                # Calculate sum of percent error and find average

                spy_average_percent_error = sum(spy_absolute_percent_error) / 12
                print("Average percent error of MSF_final on S&P 500 stock is: ", spy_average_percent_error * 100, "%")

            if df['instrumentid'][i] == 4:
                xph_temp_error = (df['close'] - df['forecastcloseprice']) / df['close']
                xph_absolute_percent_error = [abs(ele) for ele in xph_temp_error]

                # Calculate sum of percent error and find average

                xph_average_percent_error = sum(xph_absolute_percent_error) / 12
                print("Average percent error of MSF2 on XPH stock is: ", xph_average_percent_error * 100, "%")

            if df['instrumentid'][i] == 5:
                carz_temp_error = (df['close'] - df['forecastcloseprice']) / df['close']
                carz_absolute_percent_error = [abs(ele) for ele in carz_temp_error]

                # Calculate sum of percent error and find average

                carz_average_percent_error = sum(carz_absolute_percent_error) / 12
                print("Average percent error of MSF_final on CARZ index stock is: ", carz_average_percent_error * 100,
                      "%")

            if df['instrumentid'][i] == 6:
                tyx_temp_error = (df['close'] - df['forecastcloseprice']) / df['close']
                tyx_absolute_percent_error = [abs(ele) for ele in tyx_temp_error]

                # Calculate sum of percent error and find average

                tyx_average_percent_error = sum(tyx_absolute_percent_error) / 12
                print("Average percent error of MSF_final on TYX 30-YR bond is: ", tyx_average_percent_error * 100, "%")

            if df['instrumentid'][i] == 7:
                fcau_temp_error = (df['close'] - df['forecastcloseprice']) / df['close']
                fcau_absolute_percent_error = [abs(ele) for ele in fcau_temp_error]

                # Calculate sum of percent error and find average

                fcau_average_percent_error = sum(fcau_absolute_percent_error) / 12
                print("Average percent error of MSF_final on FCAU bond is: ", fcau_average_percent_error * 100, "%")

            if df['instrumentid'][i] == 8:
                tm_temp_error = (df['close'] - df['forecastcloseprice']) / df['close']
                tm_absolute_percent_error = [abs(ele) for ele in tm_temp_error]

                # Calculate sum of percent error and find average

                tm_average_percent_error = sum(tm_absolute_percent_error) / 12
                print("Average percent error of MSF_final on Toyota Motors: ", tm_average_percent_error * 100, "%")

            if df['instrumentid'][i] == 9:
                ford_temp_error = (df['close'] - df['forecastcloseprice']) / df['close']
                ford_absolute_percent_error = [abs(ele) for ele in ford_temp_error]

                # Calculate sum of percent error and find average

                ford_average_percent_error = sum(ford_absolute_percent_error) / 12
                print("Average percent error of MSF_final on Ford stock: ", ford_average_percent_error * 100, "%")

            if df['instrumentid'][i] == 10:
                hmc_temp_error = (df['close'] - df['forecastcloseprice']) / df['close']
                hmc_absolute_percent_error = [abs(ele) for ele in hmc_temp_error]

                # Calculate sum of percent error and find average

                hmc_average_percent_error = sum(hmc_absolute_percent_error) / 12
                print("Average percent error of MSF_final on HMC: ", hmc_average_percent_error * 100, "%")

            d = len(df)
            b = (count / d) * 100
            # Prints the trend accuracy
            # print('The accuracy for instrument %d: %.2f%%\n' % (i, b))

    # End of MSF_final accuracy tests

    # Create weightings for MSF_final algorithm for past dates and compares them to actual instrument prices, generating a percent error calculation
    # We then iterate through several different weightings and we compare each percent error for each instrument and determine the weightings with the lowest percent error
    def create_weightings_MSF_final(self, setWeightings):

        # Query to grab the macroeconcodes and macroeconnames from the macroeconmaster database table
        query = "SELECT macroeconcode, macroeconname FROM dbo_macroeconmaster WHERE activecode = 'A'"
        data = pd.read_sql_query(query, self.engine)

        # Query to grab the instrumentid and instrument name from the instrumentmaster database table
        # New instruments added SS2020 so, total instruments are 10 now
        query = 'SELECT instrumentid, instrumentname FROM dbo_instrumentmaster'
        data1 = pd.read_sql_query(query, self.engine)

        # Keys is a dictionary that will be used to store the macro econ code for each macro econ name
        keys = {}
        for i in range(len(data)):
            keys.update({data['macroeconname'].iloc[i]: data['macroeconcode'].iloc[i]})

        # ikeys is a dictionary that will be used to store instrument ids for each instrument name
        ikeys = {}
        for x in range(len(data1)):
            ikeys.update({data1['instrumentname'].iloc[x]: data1['instrumentid'].iloc[x]})

        # Vars is a dictionary used to store the macro economic variable percent change for each macro economic code
        vars = {}
        # Vars is only populated with the relevant macro economic variables (GDP, COVI, UR, IR, MI)
        for i in data['macroeconcode']:
            if (i == 'GDP' or i == 'UR' or i == 'IR' or i == 'MI' or i == 'COVI'):
                d = {i: []}
                vars.update(d)

        # Weightings is used to store the best weightings for each instrument id which is returned to dataforecast and used for actual prediction
        weightings = {}

        # n represents the number of datapoints we are working with (represented in quarters)
        n = 12

        # These are the date ranges we are working with
        # start_date represents the starting date for the forecasts and the end of the training dates
        start_date = "'2018-01-01'"
        # end_date represents the date for which the forecasting ends
        end_date = "'2020-01-01'"
        # train_date represents the date we start collecting the instrument statistics used to forecast prices
        train_date = "'2015-01-01'"

        # For loop to loop through the macroeconomic codes to calculate the macro economic variable percent change
        for i in keys:
            # Check to make sure the macroeconcode we are working with is one of the relevant ones
            if keys[i] in vars:
                # Query to grab the macroeconomic statistics from the database using the relevant macro economic codes
                query = "SELECT date, statistics, macroeconcode FROM dbo_macroeconstatistics WHERE macroeconcode = {} AND date <= {}".format(
                    '"' + keys[i] + '"', start_date)
                data = pd.read_sql_query(query, self.engine)

                # For loop to retrieve macro statistics and calculate percent change
                for j in range(n):
                    # This will grab the n+1 statistic to use to calculate the percent change to the n statistic
                    temp = data.tail(n + 1)
                    # This will grab the most recent n statistics from the query, as we are working only with n points
                    data = data.tail(n)

                    # For the first iteration we need to use the n+1th statistic to calculate percent change on the oldest point
                    if j == 0:
                        macrov = (data['statistics'].iloc[j] - temp['statistics'].iloc[0]) / temp['statistics'].iloc[0]
                        vars[keys[i]].append(macrov)
                    else:
                        macrov = (data['statistics'].iloc[j] - data['statistics'].iloc[j - 1]) / \
                                 data['statistics'].iloc[j - 1]
                        vars[keys[i]].append(macrov)

        # If you are not using set weightings then this if statement will run and create the best fit weightings
        if not setWeightings:
            # We now iterate through the instrument ids
            for x in ikeys:

                # This query will grab the quarterly instrument statistics from 2016 to 2018
                query = "SELECT date, close, instrumentid FROM ( SELECT date, close, instrumentid, ROW_NUMBER() OVER " \
                        "(PARTITION BY YEAR(date), MONTH(date) ORDER BY DAY(date) DESC) AS rowNum FROM " \
                        "dbo_instrumentstatistics WHERE instrumentid = {} AND date BETWEEN {} AND {} ) z " \
                        "WHERE rowNum = 1 AND ( MONTH(z.date) = 3 OR MONTH(z.date) = 6 OR MONTH(z.date) = 9 OR " \
                        "MONTH(z.date) = 12)".format(ikeys[x], train_date, start_date)

                # Then we execute the query and store the returned values in instrumentStats, and grab the last n stats from the dataframe as we are only using n datapoints
                instrumentStats = pd.read_sql_query(query, self.engine)
                instrumentStats = instrumentStats.tail(n)

                # Best weightings will be used to store the best weightings for each instrument
                best_weightings = [0, 0, 0]

                # Best avg error will be used to store the best average percent error for each isntrument
                best_avg_error = -1

                # Best trend error will be used to store the best trend error for each instrument
                best_trend_error = -1

                # Best forecast prices will be used to store the forecast prices for the best weightings to store them in a database for visual comparison later
                best_forecast_prices = []

                # We now iterate through all 3 different possible weightings
                for weight in numpy.arange(-5.7, 2.8, .25):
                    for uweight in numpy.arange(-3.7, 3.6, .25):
                        for iweight in numpy.arange(-.8, .9, .25):

                            # We intialize a list to store the resulting forecasted prices to compare in another function
                            stat_check = []

                            # isFirst will determine whether or not this is the first calculation being done
                            # If it is true then we use the most recent instrument statistic to forecast the first pricepoint
                            # IF it is false then we use the previous forecast price to predict the next forecast price
                            isFirst = True

                            # This is the actual calculation of MSF3 where we store the result in stat_check to compare to actual instrument prices
                            for i in range(n):
                                if isFirst:
                                    # Change to pluses and test accuracy
                                    stat = vars['GDP'][i] * weight - vars['UR'][i] * uweight + vars['IR'][
                                        i] * iweight - (
                                                   vars['MI'][i] * vars['MI'][i])
                                    stat = (stat * instrumentStats['close'].iloc[n - 1]) + \
                                           instrumentStats['close'].iloc[n - 1]
                                    stat_check.append(stat)
                                    temp_price = stat
                                    isFirst = False
                                else:
                                    stat = vars['GDP'][i] * weight - (
                                                vars['UR'][i] * uweight + vars['IR'][i] * iweight) - (
                                                   vars['MI'][i] * vars['MI'][i])
                                    stat = (stat * temp_price) + temp_price
                                    stat_check.append(stat)
                                    temp_price = stat

                            # We call to the weight check function using the list of forecasted prices, the current instrument id, the amount of datapoints we are working with, and the name of the function we are testing
                            # It then returns the average percent error and trend error for the forecasted prices, as well as the dates we are forecasting for so we can insert them into the visualize table
                            temp_avg_error, temp_trend_error, dates = weight_check(DBEngine().mysql_engine(),
                                                                                   stat_check, ikeys[x], n, 'MSF_final',
                                                                                   start_date, end_date)

                            # Check to see if the best_avg_error has been initialized to a valid average percent error, if not then no average error or trend error has been calculated yet
                            if (best_avg_error < 0):
                                # If so store the average percent error, the best weightings, best trend error, and the resulting forecasted prices for comparison with other weightings
                                best_avg_error = temp_avg_error
                                best_weightings = [weight, uweight, iweight]
                                best_trend_error = temp_trend_error
                                best_forecast_prices = stat_check

                            # Otherwise check if the newly calculated average percent error is worse than the newly calculated one
                            elif (best_avg_error > temp_avg_error):
                                # And if so set the values for all the relevant variables
                                best_avg_error = temp_avg_error
                                best_weightings = [weight, uweight, iweight]
                                best_trend_error = temp_trend_error
                                best_forecast_prices = stat_check

                # Print statements to view the average percent error, trend error, and best weightings
                print("The lowest avg percent error is %.7f%% for instrumentID %d" % (best_avg_error * 100, ikeys[x]),
                      ' for function: MSF2')
                print("The weightings are: ", best_weightings, ' for function: MSF_final')
                print('The trend accuracy is: ', best_trend_error)

                # initializes weightings dictionary as the best weightings found for each instrument id
                weightings[ikeys[x]] = best_weightings

                # visual_comparisons will be used to store the past forecasted prices so we can visualize them compared to actual instrument prices on a graph
                visual_comparisons = []
                for k in range(n):
                    visual_comparisons.append([dates[k], ikeys[x], best_forecast_prices[k], 'MSF_final'])
                df1 = pd.DataFrame(visual_comparisons,
                                   columns=['forecastdate', 'instrumentid', 'forecastcloseprice', 'algorithmcode'])
                df1.to_sql('dbo_tempvisualize', self.engine,
                           if_exists=('append'), index=False)

            # The weightings for each instrument ID are returned to MSF_final algorithm above and used for prediction
            return weightings

        # This else statement will make use of the preset weightings for prediction and comparison
        else:
            # These are the set weightings as of 7/28/2020, based on the added new instruments as well.
            weightings = {1: [-2.2, 3.3, 0.44999999999999996],
                          2: [1.0499999999999998, -3.2, -0.8],
                          3: [2.55, 3.3, 0.7],
                          4: [0.04999999999999982, 3.05, 0.7],
                          5: [-4.7, 3.3, 0.44999999999999996],
                          6: [-1.2000000000000002, -3.7, -0.8],
                          7: [2.55, 3.3, 0.7],
                          8: [0.04999999999999982, 3.05, 0.7],
                          9: [-4.7, 3.3, 0.44999999999999996],
                          10: [-1.2000000000000002, -3.7, -0.8]
                          }

            # We now iterate through the instrument ids
            for x in ikeys:

                # This query will grab the quarterly instrument statistics from 2016 to 2018
                query = "SELECT date, close, instrumentid FROM ( SELECT date, close, instrumentid, ROW_NUMBER() OVER " \
                        "(PARTITION BY YEAR(date), MONTH(date) ORDER BY DAY(date) DESC) AS rowNum FROM " \
                        "dbo_instrumentstatistics WHERE instrumentid = {} AND date BETWEEN {} AND {} ) z " \
                        "WHERE rowNum = 1 AND ( MONTH(z.date) = 3 OR MONTH(z.date) = 6 OR MONTH(z.date) = 9 OR " \
                        "MONTH(z.date) = 12)".format(ikeys[x], train_date, start_date)

                # Then we execute the query and store the returned values in instrumentStats, and grab the last n stats from the dataframe as we are only using n datapoints
                instrumentStats = pd.read_sql_query(query, self.engine)
                instrumentStats = instrumentStats.tail(n)

                # Best weightings will be used to store the best weightings for each instrument
                best_weightings = weightings[ikeys[x]]

                # avg error will be used to store the best average percent error for each isntrument
                avg_error = 0

                # trend error will be used to store the best trend error for each instrument
                trend_error = 0

                # Best forecast prices will be used to store the forecast prices for the best weightings to store them in a database for visual comparison later
                best_forecast_prices = []

                # We intialize a list to store the resulting forecasted prices to compare in another function
                stat_check = []

                # isFirst will determine whether or not this is the first calculation being done
                # If it is true then we use the most recent instrument statistic to forecast the first pricepoint
                # IF it is false then we use the previous forecast price to predict the next forecast price
                isFirst = True

                # This is the actual calculation of MSF3 where we store the result in stat_check to compare to actual instrument prices
                for i in range(n):
                    if isFirst:
                        # Change to pluses and test accuracy
                        stat = vars['GDP'][i] * best_weightings[0] - vars['UR'][i] * best_weightings[1] + vars['IR'][
                            i] * best_weightings[2] - (
                                       vars['MI'][i] * vars['MI'][i])
                        stat = (stat * instrumentStats['close'].iloc[n - 1]) + instrumentStats['close'].iloc[
                            n - 1]
                        stat_check.append(stat)
                        temp_price = stat
                        isFirst = False
                    else:
                        stat = vars['GDP'][i] * best_weightings[0] - (
                                    vars['UR'][i] * best_weightings[1] + vars['IR'][i] * best_weightings[2]) - (
                                       vars['MI'][i] * vars['MI'][i])
                        stat = (stat * temp_price) + temp_price
                        stat_check.append(stat)
                        temp_price = stat

                # We call to the weight check function using the list of forecasted prices, the current instrument id, the amount of datapoints we are working with, and the name of the function we are testing
                # It then returns the average percent error and trend error for the forecasted prices, as well as the dates we are forecasting for so we can insert them into the visualize table
                avg_error, trend_error, dates = FinsterTab.weight_check(DBEngine().mysql_engine(), stat_check,
                                                                        ikeys[x], n, 'MSF_final', start_date,
                                                                        end_date)

                # Print statements to view the average percent error, trend error, and best weightings
                print("The lowest avg percent error is %.7f%% for instrumentID %d" % (avg_error * 100, ikeys[x]),
                      ' for function: MSF_final')
                print("The weightings are: ", best_weightings, ' for function: MSF_final')
                print('The trend accuracy is: ', trend_error)

                # visual_comparisons will be used to store the past forecasted prices so we can visualize them compared to actual instrument prices on a graph
                visual_comparisons = []
                for k in range(n):
                    visual_comparisons.append([dates[k], ikeys[x], stat_check[k], 'MSF_final'])
                df1 = pd.DataFrame(visual_comparisons,
                                   columns=['forecastdate', 'instrumentid', 'forecastcloseprice', 'algorithmcode'])
                df1.to_sql('dbo_tempvisualize', self.engine,
                           if_exists=('append'), index=False)

            # The weightings for each instrument ID are returned to dataforecast and used for prediction
            return weightings

    def MSF2(self):
        # If you want to use set weightings, set this true. Otherwise set it false
        # If you set it to true then the weightings can be altered for MSF2 in AccuracyTest.py on line 647 in create_weightings_MSF2
        # Using set weightings will significantly speed up the run time of the application
        setWeightings = True

        # Query to grab the macroeconcodes and macroeconnames from the macroeconmaster database table
        query = "SELECT macroeconcode, macroeconname FROM dbo_macroeconmaster WHERE activecode = 'A'"
        data = pd.read_sql_query(query, self.engine)

        # Query to grab the instrumentid and instrument name from the instrumentmaster database table
        query = 'SELECT instrumentid, instrumentname FROM dbo_instrumentmaster limit 6'
        data1 = pd.read_sql_query(query, self.engine)

        # Keys is a dictionary that will be used to store the macro econ code for each macro econ name
        keys = {}
        for i in range(len(data)):
            keys.update({data['macroeconname'].iloc[i]: data['macroeconcode'].iloc[i]})

        # ikeys is a dictionary that will be used to store instrument ids for each instrument name
        ikeys = {}
        for x in range(len(data1)):
            ikeys.update({data1['instrumentname'].iloc[x]: data1['instrumentid'].iloc[x]})

        # Vars is a dictionary used to store the macro economic variable percent change for each macro economic code
        vars = {}
        for i in data['macroeconname']:
            # Vars is only populated with the relevant macro economic variables (GDP, UR, IR, and MI)
            if(i == 'GDP' or i == 'Unemployment Rate' or i == 'Inflation Rate' or i == 'Misery Index'):
                d = {i: []}
                vars.update(d)

        # Result will hold the resulting forecast prices for each instrument ID
        result = {}
        for i in data1['instrumentid']:
            d = {i: []}
            result.update(d)


        # Weightings are determined through a function written in accuracytest.py
        # The weightings returned are used in the calculation below
        weightings = FinsterTab.W2020.AccuracyTest.create_weightings_MSF2(self.engine, setWeightings)

        n = 8

        # Getting Dates for Future Forecast #
        # --------------------------------------------------------------------------------------------------------------#
        # Initialize the currentDate variable for use when grabbing the forecasted dates
        currentDate = datetime.today()

        # Creates a list to store future forecast dates
        date = []

        # This will set the value of count according to which month we are in, this is to avoid having past forecast dates in the list
        if (currentDate.month < 4):
            count = 0
        elif (currentDate.month < 7 and currentDate.month >= 4):
            count = 1
        elif (currentDate.month < 10 and currentDate.month >= 7):
            count = 2
        else:
            count = 3

        # Initialize a variable to the current year
        year = currentDate.year

        # Setup a for loop to loop through and append the date list with the date of the start of the next quarter
        # For loop will run n times, corresponding to amount of data points we are working with
        for i in range(n):
            # If the count is 0 then we are still in the first quarter
            if (count == 0):
                # Append the date list with corresponding quarter and year
                date.append(str(year) + "-03-" + "31")
                # Increase count so this date is not repeated for this year
                count += 1

            # Do the same for the next quarter
            elif (count == 1):
                date.append(str(year) + "-06-" + "30")
                count += 1

            # And for the next quarter
            elif (count == 2):
                date.append(str(year) + "-09-" + "30")
                count += 1

            # Until we account for the last quarter of the year
            else:
                date.append(str(year) + "-12-" + "31")
                # Where we then reinitialize count to 0
                count = 0
                # And then incrament the year for the next iterations
                year = year + 1
        # --------------------------------------------------------------------------------------------------------------#

        # reinitializes currentDate to todays date, also typecasts it to a string so it can be read by MySQL
        currentDate = str(datetime.today())
        currentDate = ("'" + currentDate + "'")

        # For loop to loop through the macroeconomic codes to calculate the macro economic variable percent change
        for i in keys:
            # Check to make sure the macroeconcode we are working with is one of the relevant ones
            if i in vars:
                # Query to grab the macroeconomic statistics from the database using the relevant macro economic codes
                query = 'SELECT date, statistics, macroeconcode FROM dbo_macroeconstatistics WHERE macroeconcode = {}'.format('"' + keys[i] + '"')
                data = pd.read_sql_query(query, self.engine)

                # For loop to retrieve macro statistics and calculate percent change
                for j in range(n):
                    # This will grab the n+1 statistic to use to calculate the percent change to the n statistic
                    temp = data.tail(n + 1)
                    # This will grab the most recent n statistics from the query, as we are working only with n points
                    data = data.tail(n)

                    # For the first iteration we need to use the n+1th statistic to calculate percent change on the oldest point
                    if j == 0:
                        macrov = (data['statistics'].iloc[j] - temp['statistics'].iloc[0]) / temp['statistics'].iloc[0]
                        vars[i].append(macrov)
                    else:
                        macrov = (data['statistics'].iloc[j] - data['statistics'].iloc[j - 1]) / \
                                 data['statistics'].iloc[j - 1]
                        vars[i].append(macrov)

        # We now iterate through the instrument ids
        for x in ikeys:

            # This query will grab the quarterly instrument statistics from 2014 to now
            query = "SELECT date, close, instrumentid FROM ( SELECT date, close, instrumentid, ROW_NUMBER() OVER " \
                    "(PARTITION BY YEAR(date), MONTH(date) ORDER BY DAY(date) DESC) AS rowNum FROM " \
                    "dbo_instrumentstatistics WHERE instrumentid = {} AND date BETWEEN '2014-03-21' AND {} ) z " \
                    "WHERE rowNum = 1 AND ( MONTH(z.date) = 3 OR MONTH(z.date) = 6 OR MONTH(z.date) = 9 OR " \
                    "MONTH(z.date) = 12)".format(ikeys[x], currentDate)

            # Then we execute the query and store the returned values in instrumentStats, and grab the last n stats from the dataframe as we are only using n datapoints
            instrumentStats = pd.read_sql_query(query, self.engine)
            instrumentStats = instrumentStats.tail(n)

            # Temp result will then store the resulting forecast prices throughout the calculation of n datapoints
            temp_result = []

            # isFirst will determine whether or not this is the first calculation being done
            # If it is true then we use the most recent instrument statistic to forecast the first pricepoint
            # IF it is false then we use the previous forecast price to predict the next forecast price
            isFirst = True
            # This for loop is where the actual calculation takes place
            for i in range(n):
                if isFirst:
                    stat = vars['GDP'][i] * weightings[ikeys[x]][0] - (vars['Unemployment Rate'][i] * weightings[ikeys[x]][1] + vars['Inflation Rate'][i] * weightings[ikeys[x]][2]) - (vars['Misery Index'][i] * vars['Misery Index'][i])
                    stat = (stat * instrumentStats['close'].iloc[n-1]) + instrumentStats['close'].iloc[n-1]
                    temp_result.append(stat)
                    temp_price = stat
                    isFirst = False
                else:
                    stat = vars['GDP'][i] * weightings[ikeys[x]][0] - (vars['Unemployment Rate'][i] * weightings[ikeys[x]][1] + vars['Inflation Rate'][i] *
                                weightings[ikeys[x]][2]) - (vars['Misery Index'][i] * vars['Misery Index'][i])
                    stat = (stat * temp_price) + temp_price
                    temp_result.append(stat)
                    temp_price = stat



            # We then append the resulting forcasted prices over n quarters to result, a dictionary where each
            # Instrument ID is mapped to n forecast prices
            result[ikeys[x]].append(temp_result)

        #Table will represent a temporary table with the data appended matching the columns of the macroeconalgorithmforecast database table
        table = []
        #This forloop will populate table[] with the correct values according to the database structure
        for i, k in result.items():
            cnt = 0
            for j in k:
                for l in range(n):
                    table.append([date[cnt], i, 'ALL', j[cnt], 'MSF2', 0])
                    cnt += 1

        #Once table is populated we then push it into the macroeconalgorithmforecast table
        table = pd.DataFrame(table, columns=['forecastdate','instrumentid' , 'macroeconcode',
                                            'forecastprice', 'algorithmcode', 'prederror'])
        table.to_sql('dbo_macroeconalgorithmforecast', self.engine, if_exists=('append'), index=False)


    def MSF3(self):
        # If you want to use set weightings, set this true. Otherwise set it false
        # If you set it to true then the weightings can be altered for MSF3 in AccuracyTest.py on line 1064 in create_weightings_MSF3
        # Using set weightings will significantly speed up the run time of the application
        setWeightings = True


        # Query to grab the macroeconcodes and macroeconnames from the macroeconmaster database table
        query = "SELECT macroeconcode, macroeconname FROM dbo_macroeconmaster WHERE activecode = 'A'"
        data = pd.read_sql_query(query, self.engine)

        # Query to grab the instrumentid and instrument name from the instrumentmaster database table
        query = 'SELECT instrumentid, instrumentname FROM dbo_instrumentmaster LIMIT 6'
        data1 = pd.read_sql_query(query, self.engine)

        # Keys is a dictionary that will be used to store the macro econ code for each macro econ name
        keys = {}
        for i in range(len(data)):
            keys.update({data['macroeconname'].iloc[i]: data['macroeconcode'].iloc[i]})

        # ikeys is a dictionary that will be used to store instrument ids for each instrument name
        ikeys = {}
        for x in range(len(data1)):
            ikeys.update({data1['instrumentname'].iloc[x]: data1['instrumentid'].iloc[x]})

        # Vars is a dictionary used to store the macro economic variable percent change for each macro economic code
        vars = {}
        for i in data['macroeconcode']:
            # Vars is only populated with the relevant macro economic variables (GDP, COVI, CPIUC, and FSI)
            if(i == 'GDP' or i == 'COVI' or i == 'CPIUC' or i == 'FSI'):
                d = {i: []}
                vars.update(d)

        # Result will hold the resulting forecast prices for each instrument ID
        result = {}
        for i in data1['instrumentid']:
            d = {i: []}
            result.update(d)

        # Weightings are determined through a function written in accuracytest.py
        # The weightings returned are used in the calculation below
        weightings = FinsterTab.W2020.AccuracyTest.create_weightings_MSF3(self.engine, setWeightings)


        n = 8

        # Getting Dates for Future Forecast #
        # --------------------------------------------------------------------------------------------------------------#

        # Initialize the currentDate variable for use when grabbing the forecasted dates
        currentDate = datetime.today()

        # Creates a list to store future forecast dates
        date = []

        # This will set the value of count according to which month we are in, this is to avoid having past forecast dates in the list
        if (currentDate.month < 4):
            count = 0
        elif (currentDate.month < 7 and currentDate.month >= 4):
            count = 1
        elif (currentDate.month < 10 and currentDate.month >= 7):
            count = 2
        else:
            count = 3

        # Initialize a variable to the current year
        year = currentDate.year

        # Setup a for loop to loop through and append the date list with the date of the start of the next quarter
        # For loop will run n times, corresponding to amount of data points we are working with
        for i in range(n):
            # If the count is 0 then we are still in the first quarter
            if (count == 0):
                # Append the date list with corresponding quarter and year
                date.append(str(year) + "-03-" + "31")
                # Increase count so this date is not repeated for this year
                count += 1

            # Do the same for the next quarter
            elif (count == 1):
                date.append(str(year) + "-06-" + "30")
                count += 1

            # And for the next quarter
            elif (count == 2):
                date.append(str(year) + "-09-" + "30")
                count += 1

            # Until we account for the last quarter of the year
            else:
                date.append(str(year) + "-12-" + "31")
                # Where we then reinitialize count to 0
                count = 0
                # And then incrament the year for the next iterations
                year = year + 1
        # --------------------------------------------------------------------------------------------------------------#

        # reinitializes currentDate to todays date, also typecasts it to a string so it can be read by MySQL
        currentDate = str(datetime.today())
        currentDate = ("'" + currentDate + "'")

        # For loop to loop through the macroeconomic codes to calculate the macro economic variable percent change
        for i in keys:
            # Check to make sure the macroeconcode we are working with is one of the relevant ones
            if keys[i] in vars:
                # Query to grab the macroeconomic statistics from the database using the relevant macro economic codes
                query = 'SELECT date, statistics, macroeconcode FROM dbo_macroeconstatistics WHERE macroeconcode = {}'.format(
                    '"' + keys[i] + '"')
                data = pd.read_sql_query(query, self.engine)

                # For loop to retrieve macro statistics and calculate percent change
                for j in range(n):
                    # This will grab the n+1 statistic to use to calculate the percent change to the n statistic
                    temp = data.tail(n + 1)
                    # This will grab the most recent n statistics from the query, as we are working only with n points
                    data = data.tail(n)

                    # For the first iteration we need to use the n+1th statistic to calculate percent change on the oldest point
                    if j == 0:
                        macrov = (data['statistics'].iloc[j] - temp['statistics'].iloc[0]) / temp['statistics'].iloc[0]
                        vars[keys[i]].append(macrov)
                    else:
                        macrov = (data['statistics'].iloc[j] - data['statistics'].iloc[j - 1]) / \
                                 data['statistics'].iloc[j - 1]
                        vars[keys[i]].append(macrov)

        # We now iterate through the instrument ids
        for x in ikeys:

            # This query will grab the quarterly instrument statistics from 2014 to now
            query = "SELECT date, close, instrumentid FROM ( SELECT date, close, instrumentid, ROW_NUMBER() OVER " \
                    "(PARTITION BY YEAR(date), MONTH(date) ORDER BY DAY(date) DESC) AS rowNum FROM " \
                    "dbo_instrumentstatistics WHERE instrumentid = {} AND date BETWEEN '2014-03-21' AND {} ) z " \
                    "WHERE rowNum = 1 AND ( MONTH(z.date) = 3 OR MONTH(z.date) = 6 OR MONTH(z.date) = 9 OR " \
                    "MONTH(z.date) = 12)".format(ikeys[x], currentDate)

            # Then we execute the query and store the returned values in instrumentStats, and grab the last n stats from the dataframe as we are only using n datapoints
            instrumentStats = pd.read_sql_query(query, self.engine)
            instrumentStats = instrumentStats.tail(n)

            # Temp result will then store the resulting forecast prices throughout the calculation of n datapoints
            temp_result = []

            # isFirst will determine whether or not this is the first calculation being done
            # If it is true then we use the most recent instrument statistic to forecast the first pricepoint
            # IF it is false then we use the previous forecast price to predict the next forecast price
            isFirst = True
            # This for loop is where the actual calculation takes place
            for i in range(n):
                if isFirst:
                    stat = vars['GDP'][i] * weightings[ikeys[x]][0] - (vars['COVI'][i] * weightings[ikeys[x]][1] + vars['FSI'][i] * weightings[ikeys[x]][2]) - \
                           (vars['CPIUC'][i] * vars['CPIUC'][i])
                    stat = (stat * instrumentStats['close'].iloc[n-1]) + instrumentStats['close'].iloc[n-1]
                    temp_result.append(stat)
                    temp_price = stat
                    isFirst = False
                else:
                    stat = vars['GDP'][i] * weightings[ikeys[x]][0] - (vars['COVI'][i] * weightings[ikeys[x]][1] + vars['FSI'][i] * weightings[ikeys[x]][2]) - \
                           (vars['CPIUC'][i] * vars['CPIUC'][i])
                    stat = (stat * temp_price) + temp_price
                    temp_result.append(stat)
                    temp_price = stat


            # We then append the resulting forcasted prices over n quarters to result, a dictionary where each
            # Instrument ID is mapped to n forecast prices
            result[ikeys[x]].append(temp_result)

        # Table will represent a temporary table with the data appended matching the columns of the macroeconalgorithmforecast database table
        table = []
        # This forloop will populate table[] with the correct values according to the database structure
        for i, k in result.items():
            cnt = 0
            for j in k:
                for l in range(n):
                    table.append([date[cnt], i, 'ALL', j[cnt], 'MSF3', 0])
                    cnt += 1

        # Once table is populated we then push it into the macroeconalgorithmforecast table
        table = pd.DataFrame(table, columns=['forecastdate','instrumentid' , 'macroeconcode',
                                            'forecastprice', 'algorithmcode', 'prederror'])
        table.to_sql('dbo_macroeconalgorithmforecast', self.engine, if_exists=('append'), index=False)

    def MSF2_Past_Date(self):

        # Query to grab the macroeconcodes and macroeconnames from the macroeconmaster database table
        query = "SELECT macroeconcode, macroeconname FROM dbo_macroeconmaster WHERE activecode = 'A'"
        data = pd.read_sql_query(query, self.engine)

        # Query to grab the instrumentid and instrument name from the instrumentmaster database table
        query = 'SELECT instrumentid, instrumentname FROM dbo_instrumentmaster WHERE instrumentid = 3'
        data1 = pd.read_sql_query(query, self.engine)

        # Keys is a dictionary that will be used to store the macro econ code for each macro econ name
        keys = {}
        for i in range(len(data)):
            keys.update({data['macroeconname'].iloc[i]: data['macroeconcode'].iloc[i]})

        # ikeys is a dictionary that will be used to store instrument ids for each instrument name
        ikeys = {}
        for x in range(len(data1)):
            ikeys.update({data1['instrumentname'].iloc[x]: data1['instrumentid'].iloc[x]})

        # Vars is a dictionary used to store the macro economic variable percent change for each macro economic code
        vars = {}
        for i in data['macroeconname']:
            # Vars is only populated with the relevant macro economic variables (GDP, UR, IR, and MI)
            if(i == 'GDP' or i == 'Unemployment Rate' or i == 'Inflation Rate' or i == 'Misery Index'):
                d = {i: []}
                vars.update(d)

        # Result will hold the resulting forecast prices for each instrument ID
        result = {}
        for i in data1['instrumentid']:
            d = {i: []}
            result.update(d)


        # Weightings are determined through a function written in accuracytest.py
        # The weightings returned are used in the calculation below
        weightings = FinsterTab.W2020.AccuracyTest.create_weightings_MSF2_Past_Dates(self.engine)

        n = 8

        # Getting Dates for Future Forecast #
        # --------------------------------------------------------------------------------------------------------------#
        # Initialize the currentDate variable for use when grabbing the forecasted dates
        currentDate = datetime.today()

        # Creates a list to store future forecast dates
        date = []

        # This will set the value of count according to which month we are in, this is to avoid having past forecast dates in the list
        if (currentDate.month < 4):
            count = 0
        elif (currentDate.month < 7 and currentDate.month >= 4):
            count = 1
        elif (currentDate.month < 10 and currentDate.month >= 7):
            count = 2
        else:
            count = 3

        # Initialize a variable to the current year
        year = currentDate.year

        # Setup a for loop to loop through and append the date list with the date of the start of the next quarter
        # For loop will run n times, corresponding to amount of data points we are working with
        for i in range(n):
            # If the count is 0 then we are still in the first quarter
            if (count == 0):
                # Append the date list with corresponding quarter and year
                date.append(str(year) + "-03-" + "31")
                # Increase count so this date is not repeated for this year
                count += 1

            # Do the same for the next quarter
            elif (count == 1):
                date.append(str(year) + "-06-" + "30")
                count += 1

            # And for the next quarter
            elif (count == 2):
                date.append(str(year) + "-09-" + "30")
                count += 1

            # Until we account for the last quarter of the year
            else:
                date.append(str(year) + "-12-" + "31")
                # Where we then reinitialize count to 0
                count = 0
                # And then incrament the year for the next iterations
                year = year + 1
        # --------------------------------------------------------------------------------------------------------------#

        # reinitializes currentDate to todays date, also typecasts it to a string so it can be read by MySQL
        currentDate = str(datetime.today())
        currentDate = ("'" + currentDate + "'")

        # For loop to loop through the macroeconomic codes to calculate the macro economic variable percent change
        for i in keys:
            # Check to make sure the macroeconcode we are working with is one of the relevant ones
            if i in vars:
                # Query to grab the macroeconomic statistics from the database using the relevant macro economic codes
                query = 'SELECT date, statistics, macroeconcode FROM dbo_macroeconstatistics WHERE macroeconcode = {}'.format('"' + keys[i] + '"')
                data = pd.read_sql_query(query, self.engine)

                # For loop to retrieve macro statistics and calculate percent change
                for j in range(n):
                    # This will grab the n+1 statistic to use to calculate the percent change to the n statistic
                    temp = data.tail(n + 1)
                    # This will grab the most recent n statistics from the query, as we are working only with n points
                    data = data.tail(n)

                    # For the first iteration we need to use the n+1th statistic to calculate percent change on the oldest point
                    if j == 0:
                        macrov = (data['statistics'].iloc[j] - temp['statistics'].iloc[0]) / temp['statistics'].iloc[0]
                        vars[i].append(macrov)
                    else:
                        macrov = (data['statistics'].iloc[j] - data['statistics'].iloc[j - 1]) / \
                                 data['statistics'].iloc[j - 1]
                        vars[i].append(macrov)

        # We now iterate through the instrument ids
        for x in ikeys:

            # This query will grab the quarterly instrument statistics from 2014 to now
            query = "SELECT date, close, instrumentid FROM ( SELECT date, close, instrumentid, ROW_NUMBER() OVER " \
                    "(PARTITION BY YEAR(date), MONTH(date) ORDER BY DAY(date) DESC) AS rowNum FROM " \
                    "dbo_instrumentstatistics WHERE instrumentid = {} AND date BETWEEN '2014-03-21' AND {} ) z " \
                    "WHERE rowNum = 1 AND ( MONTH(z.date) = 3 OR MONTH(z.date) = 6 OR MONTH(z.date) = 9 OR " \
                    "MONTH(z.date) = 12)".format(3, currentDate)

            # Then we execute the query and store the returned values in instrumentStats, and grab the last n stats from the dataframe as we are only using n datapoints
            instrumentStats = pd.read_sql_query(query, self.engine)
            instrumentStats = instrumentStats.tail(n)

            # Temp result will then store the resulting forecast prices throughout the calculation of n datapoints
            temp_result = []

            # isFirst will determine whether or not this is the first calculation being done
            # If it is true then we use the most recent instrument statistic to forecast the first pricepoint
            # IF it is false then we use the previous forecast price to predict the next forecast price
            isFirst = True
            # This for loop is where the actual calculation takes place
            for i in range(n):
                if isFirst:
                    stat = vars['GDP'][i] * weightings[ikeys[x]][0] - (vars['Unemployment Rate'][i] * weightings[ikeys[x]][1] + vars['Inflation Rate'][i] * weightings[ikeys[x]][2]) - (vars['Misery Index'][i] * vars['Misery Index'][i])
                    stat = (stat * instrumentStats['close'].iloc[n-1]) + instrumentStats['close'].iloc[n-1]
                    temp_result.append(stat)
                    temp_price = stat
                    isFirst = False
                else:
                    stat = vars['GDP'][i] * weightings[ikeys[x]][0] - (vars['Unemployment Rate'][i] * weightings[ikeys[x]][1] + vars['Inflation Rate'][i] *
                                weightings[ikeys[x]][2]) - (vars['Misery Index'][i] * vars['Misery Index'][i])
                    stat = (stat * temp_price) + temp_price
                    temp_result.append(stat)
                    temp_price = stat



            # We then append the resulting forcasted prices over n quarters to result, a dictionary where each
            # Instrument ID is mapped to n forecast prices
            result[ikeys[x]].append(temp_result)

        #Table will represent a temporary table with the data appended matching the columns of the macroeconalgorithmforecast database table
        table = []

        #This forloop will populate table[] with the correct values according to the database structure
        for i, k in result.items():
            cnt = 0
            for j in k:
                for l in range(n):
                    table.append([date[cnt], i, 'ALL', j[cnt], 'MSF2 Past Dates', 0])
                    cnt += 1

        #Once table is populated we then push it into the macroeconalgorithmforecast table
        table = pd.DataFrame(table, columns=['forecastdate','instrumentid' , 'macroeconcode',
                                            'forecastprice', 'algorithmcode', 'prederror'])
        table.to_sql('dbo_macroeconalgorithmforecast', self.engine, if_exists=('append'), index=False)


    # Calculation function used in MSF1
    def calc(self, df1, df2, n):
        G = 0

        # Calculates average Macro Variable % change over past n days
        for i in range(n):
            G = df1['statistics'][i] + G

        G = G / n
        G = G / 100
        return G




# END CODE MODULE