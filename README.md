# Anomaly Detection and Forecasting on Credit Card Transactions / Stock Prices

This is my personal project to learn about training and evaluating ML models for anomaly / outlier detection and forecasting.  
The dataset used is a simulated dataset found [here](https://github.com/Fraud-Detection-Handbook/simulated-data-raw).  
I refer to the documentation [here](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/Introduction.html) to get better clarity on the dataset and problem statement.

I train and evaluate anomaly detection on the following models:
* Isolation Forest (using H2O)
* KMeans (using H2O)
* Local Outlier Factor
* One Class SVM
* Autoencoders (using H2O)

I also use SHAP to provide explainability to each model to understand how each models determines which data points are anomalous.

I also perform forecasting using following models to predict TESLA stock prices for a given future period.
* Autoregression
* ARIMA
* XGBoost