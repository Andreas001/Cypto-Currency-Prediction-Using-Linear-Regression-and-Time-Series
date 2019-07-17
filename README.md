<h1 align="center">
    Crypto Currency Trend Prediction
</h1>

## Getting started

This project uses python

These libraries are used in the project:
- Tensorflow
- Keras
- Sklearn
- Pandas
- Matplotlib
- Numpy

# How to use

Do this

## Code

```py
data = pd.read_csv('BCH-USD.csv', header=0)  # load data set
X = data.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
Y = data.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions
```

```py
plt.plot(X, Y, color='blue') # actual data
plt.plot(X, Y_pred, color='red') # linear regression line
plt.show()
```

![screenshot-linear-regression](https://raw.githubusercontent.com/Andreas001/CryptoCurrencyTrendPrediction/master/Screenshots/CryptoBCH-USD10000.PNG)

## Authors

* **[Andreas Geraldo](https://github.com/Andreas001)**
* **[Thompson Darmawan Yanelie](https://github.com/insert-name)**
* **[Jerry Aivanca Pattikawa](https://github.com/insert-name)**
