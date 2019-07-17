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

## How to use

Do this

## Linear regression

This is the most simplest but still useful method, the code below will generate the screenshot shown below

```python
data = pd.read_csv('BCH-USD.csv', header=0)  # load data set
X = data.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
Y = data.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions
```

```python
plt.plot(X, Y, color='blue') # actual data
plt.plot(X, Y_pred, color='red') # linear regression line
plt.show()
```

![screenshot-linear-regression](https://raw.githubusercontent.com/Andreas001/CryptoCurrencyTrendPrediction/master/Screenshots/LinearRegression.PNG)

## Deep learning

```python
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)
```
```python
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
dataframe = read_csv('1880-2019.csv', usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')
```

```python
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
# reshape dataset
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
```

```python
# create and fit Multilayer Perceptron model
model = Sequential()
model.add(Dense(12, input_dim=look_back, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=60, batch_size=2, verbose=2)
# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
```

```python
# generate predictions for training
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
```

```python
# plot baseline and predictions
plt.plot(dataset)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
```

First image: Raw data
Second image: training and then actual prediction


![screenshot-linear-regression](https://raw.githubusercontent.com/Andreas001/CryptoCurrencyTrendPrediction/master/Screenshots/Dataset.PNG)


![screenshot-linear-regression](https://raw.githubusercontent.com/Andreas001/CryptoCurrencyTrendPrediction/master/Screenshots/Prediction.PNG)


## Authors

* **[Andreas Geraldo](https://github.com/Andreas001)**
* **[Thompson Darmawan Yanelie](https://github.com/insert-name)**
* **[Jerry Aivanca Pattikawa](https://github.com/insert-name)**
