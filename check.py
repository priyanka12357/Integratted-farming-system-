import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import linear_model
import os

from datetime import datetime, timedelta

def check_stat(File):
    # Split the filename using dot as delimiter
    namee = File.split(".")
    crop_n = namee[0]

    try:
        os.remove("static/result_stats.png")
    except:
        print("Graph Already Deleted or Permission Error")

    # Load the dataset
    df = pd.read_csv(f"datas//{File}")

    # Convert 'Price Date' to datetime
    df['pricedate'] = pd.to_datetime(df['pricedate'])

    # Extracting and renaming the important variables
    df['Mean'] = (df['minprice'] + df['maxprice']) / 2

    # Cleaning the data for any NaN or Null fields
    df = df.dropna()

    # Prediction mean based upon min price
    X = np.array(df['minprice'], dtype='float32')
    Y = np.array(df['Mean'], dtype='float32')

    # Splitting data into train and test
    N = 2441
    Xtrain, Xtest = X[:N], X[-272:]
    ytrain, ytest = Y[:N], Y[-272:]

    # Bayesian Ridge Regression
    reg = linear_model.BayesianRidge()
    reg.fit(Xtrain.reshape(-1, 1), ytrain)
    y_pred = reg.predict(Xtest.reshape(-1, 1))

    # Find the highest predicted price and its index
    highest_price = max(y_pred)
    highest_price_index = np.argmax(y_pred)  # Index of the highest price

    # Calculate the day corresponding to the highest price
    # start_date = df['pricedate'].iloc[-272]  # Start date of test data
    start_date = datetime.today().date()
    # highest_price_date = start_date + timedelta(days=highest_price_index)
    highest_price_date = start_date + timedelta(days=int(highest_price_index))

    # Format the result as month and year
    highest_price_month_year = highest_price_date.strftime("%B %Y")
    

    # Print the result
    print(f"The highest predicted price is {highest_price} on {highest_price_month_year}.")

    # Plot the prediction
    plt.plot(ytest, label='actual')
    plt.plot(y_pred, label='predicted')
    plt.xlabel('Days')
    plt.ylabel('Prices')
    plt.title(f'Time Series for {crop_n}')
    plt.legend()
    plt.savefig("static/result_stats.png")
    plt.close()

    return highest_price, highest_price_month_year

