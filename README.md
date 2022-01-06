# Financial Time Series Forecasting
The aim of this study is to demonstrate whether an effective forecast can be made on Stock Market movements like BIST 100 and USD/TRY parity movement using artificial neural networks and other machine learning techniques.

In this study, eight different machine learning, deep learning and artificial neural network models were used.

Considering the model evaluation criteria, after the training process of the dataset, the artificial neural network CNN/LSTM model differs from other models. According to the Model Performance Criterias, the highest Accuracy Score, the lowest RMSE, the lowest AIC and BIC values are seen in the CNN/LSTM model.

The outputs of Linear Discriminant Analysis and Quadratic Discriminant Analysis used for Linear Regression and Principal Component Analysis gave close or similar results.

The worst performance was obtained in the Decision Tree model

In addition, USD/TRY parity estimation gave better results than BIST 100 index estimation. The financial explanation for this is that the USD/TRY parity has shown more volatility and high trend in the last 5 years. While estimating the BIST 100 index, a delay of 1 year was taken to avoid the autocorrelation effect due to the fact that Borsa Istanbul changed the data keeping method as of 27 July 2020. As a result, it is seen that the extraordinary effects of the pandemic period are not reflected in the BIST 100 index forecast.

Compared to the studies in the literature, this study gives a result over 50% in the Financial Time Series estimations made using artificial neural networks.
