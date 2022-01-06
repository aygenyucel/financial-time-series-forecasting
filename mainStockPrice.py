import datetime
import numpy as np
from numpy import array
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import pandas_datareader.data as web

from math import log
from RegscorePy import *
from pandas_datareader.data import DataReader
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression as LR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.neighbors import KNeighborsClassifier as KNN

# XU100.IS = BIST100

def create_lagged_series(symbol, start_date, end_date, lags=5):
    """This creates a pandas DataFrame that stores the percentage returns of the 
    adjusted closing value of a stock obtained from Yahoo Finance, along with 
    a number of lagged returns from the prior trading days (lags defaults to 5 days).
    Trading volume, as well as the Direction from the previous day, are also included."""

    # Obtain stock information from Yahoo Finance(gecikmelerden dolayı 365 gün geriden başladık)
    ts = web.DataReader(symbol, 'yahoo', start_date-datetime.timedelta(days=365), end_date)
    print("ts: ",ts.shape)
    print(ts.head())
    print(ts.tail())
    
    # Create the new lagged DataFrame
    tslag = pd.DataFrame(index=ts.index)
    tslag["Today"] = ts["Adj Close"]
    tslag["Volume"] = ts["High"] - ts["Low"] # BIST 100 için sadece ts["Volume"] yazılacak
    # USD/TRY için tslag["Volume"] = ts["High"] - ts["Low"] yazılacak
    print("tslag: ",tslag.shape)
    print(tslag.head())
    print(tslag.tail())
    
    plot_cols = ['Today', 'Volume']
    plot_features = tslag[plot_cols]
    plot_features.index = tslag.index
    _ = plot_features.plot(subplots=True)
    plot_features = tslag[plot_cols][2904:]    # BIST100 den 2 sıfır atılma tarihi 27/07/2020 baz alınmıştır
    plot_features.index = tslag.index[2904:]
    _ = plot_features.plot(subplots=True)
    plt.show()

    # Create the shifted lag series of prior trading period close values
    for i in range(0,lags):
        tslag["Lag%s" % str(i+1)] = ts["Adj Close"].shift(i+1)

    # Create the returns DataFrame
    tsret = pd.DataFrame(index=tslag.index)
    tsret["Volume"] = tslag["Volume"]
    tsret["Today"] = tslag["Today"].pct_change()*100.0
    print("tsret: ",tsret.shape)
    print(tsret.head())
    print(tsret.tail())
    # If any of the values of percentage returns equal zero, set them to
    # a small number (stops issues with QDA model in scikit-learn)
    for i,x in enumerate(tsret["Today"]):
        if (abs(x) < 0.0001):
            tsret["Today"][i] = 0.0001

    # Create the lagged percentage returns columns
    for i in range(0,lags):
        tsret["Lag%s" % str(i+1)] = tslag["Lag%s" % str(i+1)].pct_change()*100.0

    # Create the "Direction" column (+1 or -1) indicating an up/down day
    tsret["Direction"] = np.sign(tsret["Today"])
    tsret = tsret[tsret.index >= start_date]

    return tsret

def fit_model(name, model, X_train, y_train, X_test, pred):
    """Fits a classification model (for our purposes this is LR, LDA and QDA)
    using the training data, then makes a prediction and subsequent "hit rate"
    for the test data."""

    
    # Fit and predict the model on the training, and then test, data
    model = model.fit(X_train, y_train)
    pred[name] = model.predict(X_test)

    # Create a series with 1 being correct direction, 0 being wrong
    # and then calculate the hit rate based on the actual direction
    pred["%s_Correct" % name] = (1.0+pred[name]*pred["Actual"])/2.0
    hit_rate = np.mean(pred["%s_Correct" % name])
    print("----------------------------------------------------------------")
    print ("The Hit Rate of %s: %.3f" % (name, hit_rate))

    # Calculate the accuracy
    print("The Accuracy Score %s:  %0.3f" % (name, (accuracy_score(y_test, pred["%s_Correct" % name], normalize=True)*100)))   
    # hit rate için score hesaplaması yapıldı
    print("The Hit Score of %s: %0.3f" % (name, model.score(X_test, y_test)*100))
    #target_names = ['class -1', 'class 0', 'class 1']
    # classification report 
    print("The Classification Report of %s: \n" % name)
    print(classification_report(y_test, (pred["%s_Correct" % name]), target_names=None, zero_division = 1, labels=np.unique(model.predict(X_test))))

    print('%s Mean Absolute Error: %0.3f' % (name, metrics.mean_absolute_error(y_test, pred["%s_Correct" % name])))
    print('%s Mean Squared Error: %0.3f' % (name, metrics.mean_squared_error(y_test, pred["%s_Correct" % name])))
    print('%s Root Mean Squared Error: %0.3f' % (name, np.sqrt(metrics.mean_squared_error(y_test, pred["%s_Correct" % name]))))
    # Akaike’s Information Criterion (En düşük AIC modeli seçilir)
    print("The AIC of %s: %0.3f" % (name, aic.aic(y_test, (pred["%s_Correct" % name]), len(model.predict(X_test)))))
    # Bayesian Information Criterion (En düşük BIC modeli seçilir)
    print("The BIC of %s: %0.3f\n" % (name, bic.bic(y_test, (pred["%s_Correct" % name]), len(model.predict(X_test)))))
    

# def baseline_model():
#  	  # create model
#     model = Sequential()
#     model.add(Conv1D(200, kernel_size=2))
#     model.add(LSTM(200))
#     model.add(Dense(100))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model   

if __name__ == '__main__' :
    # Create a lagged series of the USD/TRY index: TRY=X; XU100.IS = BIST100; ^XU100
    snpret = create_lagged_series("XU100.IS", datetime.datetime(2016,1,1), datetime.datetime(2020,7,24), lags=5)
    print("snpret: ",snpret.shape)
    print(snpret.head())
    print(snpret.tail())

    
    # Use the prior two days of returns as predictor values, with direction as the response
    X = snpret[["Lag1","Lag2"]]
    y = snpret["Direction"]

    # The test data is split into two parts: Before and after 1st Jan 2020.
    start_test = datetime.datetime(2019,7,24)

    #cv = KFold(n_splits=10, random_state=42, shuffle=False)


    # Create training and test sets
    X_train = X[X.index < start_test]
    X_test = X[X.index >= start_test]
    y_train = y[y.index < start_test]
    y_test = y[y.index >= start_test]

    # Create prediction DataFrame
    pred = pd.DataFrame(index=y_test.index)
    pred["Actual"] = y_test

       
    # Create and fit the three models    
    print ("Hit Rates of XU100.IS(BIST100):")
    models = [("LR", LR(max_iter=1000000, solver = 'sag', C= 1000000)), ("LDA", LDA(solver = 'lsqr')), ("QDA", QDA()), 
                        ("LSVC", LinearSVC(max_iter=1000000, C=100)),
                        ("DTC", DTC(random_state=20)),
                        ("SVC", SVC(
                                C=1000000.0, cache_size=200, class_weight=None,
                                coef0=0.0, degree=3, gamma=0.0001, kernel='rbf',
                                max_iter=-1, probability=False, random_state=None,
                                shrinking=True, tol=0.001, verbose=False)),
                        ("RFC", RFC(
                                n_estimators=1000, criterion='gini',
                                max_depth=None, min_samples_split=2,
                                min_samples_leaf=30, max_features='auto',
                                bootstrap=True, oob_score=False, n_jobs=1,
                                random_state=None, verbose=0)
                            ),
                        ("KNN", KNN(n_neighbors=5))    
                            
                            ]
    for m in models:
        fit_model(m[0], m[1], X_train, y_train, X_test, pred)
        
     
    
 




