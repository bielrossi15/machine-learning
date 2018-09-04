import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quandl
from matplotlib import style
from sklearn import model_selection, preprocessing, svm
from sklearn.linear_model import LinearRegression
import datetime

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']] #features adjusted
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0 #percent high/low change, volatility
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0 #percentual change in/out

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']] #columns in df.head

forecast_col = 'Adj. Close' #chosing the label to predict
df.fillna(-99999, inplace=True) #filling NaN errors with -99999

forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(columns = ['label'])) #X is a feature variable, so we use all, except label
X = preprocessing.scale(X) #normalization, eliminates sparsity, bringing all values onto one scale
X_lately = X[-forecast_out:] #the X that we gonna predict again
X = X[:-forecast_out]

df.dropna(inplace=True) #drop all the missing values
y = np.array(df['label']) #y is a label variable



X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2) #split arrays into random train and tests subsets

classifier = LinearRegression(n_jobs=-1) #supervised learning, clf is used to predict non used data
classifier.fit(X_train, y_train) #uses train values to train the classifier
accuracy = classifier.score(X_test, y_test) #tests the accuracy of the classifier
print(accuracy)

forecast_set = classifier.predict(X_lately)
#print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

print(df.head())


df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
