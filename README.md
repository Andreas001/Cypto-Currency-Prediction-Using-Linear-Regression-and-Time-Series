<h1 align="center">
    Crypto Currency Price Prediction
</h1>

## Getting started

This project is about taking cypto currency taken from [Crypocurrency Data](http://www.cryptodatadownload.com/data/northamerican/) saved as csv and then using that to display the data using Table, Graph, and Candlestick. But that isnt the only thing that's happening, the data is also used for simple predictions that is Linear regression & Time series where their results will also be shown by a graph.

This project uses python and you can use any program like what i use, Pycharm to run the code. Before being able to run the code you will need to install these libraries: 

- dash
- sklearn
- pandas
- plotly
- numpy
- keras

Dont forget to put the csv files in the same place as your code

Note: all code shown in this readme might not be the final code or the exact code in app.py but rather to show you how it works or if the actual code for method is all you needed.

## How to use

Once you got your setup just run the code and it should be running on your localhost, in pycharm it will give you a link that you can click in the debbugger.

#### Dropdown

You can pick any of the given options of crypto currencies and everything will update

![screenshot-dropdown](https://raw.githubusercontent.com/Andreas001/Cypto-Currency-Prediction-Using-Linear-Regression-and-Time-Series/master/screenshots/Dropdown.png)

#### Table

A table will show you all of the data contained in the csv provided. You will be able to go through every single column and row with the table.

![screenshot-table](https://raw.githubusercontent.com/Andreas001/Cypto-Currency-Prediction-Using-Linear-Regression-and-Time-Series/master/screenshots/Table.png)

#### Graph

This graph will show you a line graph of the actual data.

![screenshot-table](https://raw.githubusercontent.com/Andreas001/Cypto-Currency-Prediction-Using-Linear-Regression-and-Time-Series/master/screenshots/Graph.png)

#### Candlestick

Since this is crypto currency so showing it with a candlestick would be optimal to show the complete actual data.

![screenshot-table](https://raw.githubusercontent.com/Andreas001/Cypto-Currency-Prediction-Using-Linear-Regression-and-Time-Series/master/screenshots/Candlestick.png)

#### Linear regression

This is the most simplest and probably every data researcher first method in data science

This code creates the prediction

```python
data = pd.read_csv('BCH-USD.csv', header=0)  # load data set
X = data.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
Y = data.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions
```

This code will give you a graph incase you wanted a quick test, this code isn't in app.py

```python
plt.plot(X, Y, color='blue') # actual data
plt.plot(X, Y_pred, color='red') # linear regression line
plt.show()
```

![screenshot-linear-regression](https://raw.githubusercontent.com/Andreas001/Cypto-Currency-Prediction-Using-Linear-Regression-and-Time-Series/master/screenshots/Linear_Regression.png)

## Time Series

See screenshot after code to see what it produce

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

![screenshot-linear-regression](https://raw.githubusercontent.com/Andreas001/Cypto-Currency-Prediction-Using-Linear-Regression-and-Time-Series/master/screenshots/Time_Series.png)

## Authors

* **[Andreas Geraldo](https://github.com/Andreas001)**
* **[Thompson Darmawan Yanelie](https://github.com/insert-name)**
* **[Jerry Aivanca Pattikawa](https://github.com/insert-name)**

## Acknowledgments

Crypto currency data - [cryptodatadownload](http://www.cryptodatadownload.com/data/northamerican/)

Linear regression - [towardsdatascience](https://towardsdatascience.com/linear-regression-in-6-lines-of-python-5e1d0cd05b8d)

Time series code - [machinelearningmastery](https://machinelearningmastery.com/time-series-prediction-with-deep-learning-in-python-with-keras/)

