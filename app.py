import dash
import dash_table
import datetime
import dash_core_components as dcc
import dash_html_components as html
from sklearn.linear_model import LinearRegression
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objs as go
import numpy
from keras.models import Sequential
from keras.layers import Dense
import math

available_indicators = ['Linear Regression', 'Deep Learning', 'Machine Learning']
df = pd.read_csv('Gemini_BTCUSD_daily.csv')
dft = pd.read_csv('Gemini_ETHUSD_daily.csv')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([

    html.Div([

        # Title
        html.Div([
          html.H1(children='Crypto Currency Prediction')
        ], style={'textAlign': 'center'}),

        # Drop down and data table
        html.Div([
            html.H3(children='Crypto Currency Type'),

            dcc.Dropdown(
                id='table-dropdown',
                options=[
                    {'label': 'Bitcoin', 'value': 'BTC'},
                    {'label': 'Ethirium', 'value': 'ETH'}
                ],
                value='BTC'
            ),

            dcc.Markdown('''
            #### Choose one of the crypto currency
            
            Historical data of crypto currency daily price including it's highest, lowest, open, close, etc.
            All data is taken from [Crypto Data](http://www.cryptodatadownload.com/data/northamerican/).
            Data will be shown in both a table and graph.
            
            '''),

            html.H3(children='Data Table'),

            dash_table.DataTable(
                id='table',
                data=df.to_dict('records'),
                columns=[{'id': c, 'name': c} for c in df.columns],
                fixed_rows={'headers': True, 'data': 0},
                style_cell={'width': '150px'}
            )],
        ),

        html.Div([
            html.H3(children='Data Graph'),

            dcc.Graph(id='data-graph')
        ]),

        html.Div([
            html.H3(children='Candlestick'),

            dcc.Graph(id='data-candlestick')
        ]),

        html.Div([
            html.H2(children='Prediction using linear regression'),

            dcc.Markdown('''
            #### Why linear regression ?

            Linear regression is one of the simplest model and not the main method that will be used in this prediction.
            It is only to being used as a learning experience for the team members and possible others.

            '''),

            dcc.Graph(id='graph')
        ]),

        html.Div([
            html.H2(children='Time series prediction'),

            dcc.Markdown('''
            #### Time series prediction with deep learning

            Please wait while the algorithm works with the data, it might take a moment.
            This method is not ours and the original can be found in 
            [Time series with deep learning](https://machinelearningmastery.com/time-series-prediction-with-deep-learning-in-python-with-keras/)

            '''),

            dcc.Graph(id='graph-learn')
        ]),

    ]),
])


@app.callback(Output('table', 'data'), [Input('table-dropdown', 'value')])
def update_rows(value):
    # dff = df[df['some-column'] == value]
    if value == 'BTC':
        return df.to_dict('records')
    elif value == 'ETH':
        return dft.to_dict('records')


@app.callback(Output('data-graph', 'figure'), [Input('table-dropdown', 'value')])
def update_data_graph(value_dropdown):

    if value_dropdown == 'BTC':
        df_graph = df
        text = 'Bitcoin'
    elif value_dropdown == 'ETH':
        df_graph = dft
        text = 'Ethirium'

    return {
        'data': [go.Scatter(
            mode='lines',
            x=df_graph['Date'],
            y=df_graph['Open'],
            text=text,
            marker={
                'size': 15,
                'opacity': 0.5,
                'line': {'width': 0.5, 'color': 'blue'}
            }
        )],
        'layout': go.Layout(
            xaxis={
                'title': 'Date',
                'tickvals': [df_graph["Date"].iloc[0], df_graph["Date"].iloc[250], df_graph["Date"].iloc[500],
                             df_graph["Date"].iloc[750], df_graph["Date"].iloc[1000], df_graph["Date"].iloc[-1]],
                # 'tickformat': '%m/%y'
            },
            yaxis={
                'title': 'Price : USD',
            },
            margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
            hovermode='closest'
        )
    }


@app.callback(Output('data-candlestick', 'figure'), [Input('table-dropdown', 'value')])
def update_graph(value_dropdown):
    if value_dropdown == 'BTC':
        df_to_candlestick = df
    elif value_dropdown == 'ETH':
        df_to_candlestick = dft
    return {
        'data': [go.Candlestick(
            x=df_to_candlestick['Date'],
            open=df_to_candlestick['Open'],
            high=df_to_candlestick['High'],
            low=df_to_candlestick['Low'],
            close=df_to_candlestick['Close'],
        )],
    }


def datetime_to_float(d):
    epoch = datetime.datetime.utcfromtimestamp(0)
    total_seconds =  (d - epoch).total_seconds()
    # total_seconds will be in decimals (millisecond precision)
    return total_seconds


@app.callback(Output('graph', 'figure'), [Input('table-dropdown', 'value')])
def update_graph(value_dropdown):
    if value_dropdown == 'BTC':
        df_to_graph = df
        df_graph = df
        text = 'Bitcoin'
        X = df.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
        Y = df.iloc[:, 4].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    elif value_dropdown == 'ETH':
        df_to_graph = dft
        df_graph = dft
        text = 'Ethirium'
        X = dft.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
        Y = dft.iloc[:, 4].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column

    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions

    df_for_graph = pd.DataFrame(Y_pred, columns=['Prediction'])

    return {
        'data': [go.Scatter(
            name='Linear Regression Prediction',
            mode='lines',
            x=df_to_graph['Date'],
            y=df_for_graph['Prediction'],
            text='Linear Regression Line',
            marker={
                'size': 15,
                'opacity': 0.5,
                'line': {'width': 0.5, 'color': 'blue'}
            }
        ),
            go.Scatter(
                name='Actual Data',
                mode='lines',
                x=df_graph['Date'],
                y=df_graph['Open'],
                text=text,
                marker={
                    'size': 15,
                    'opacity': 0.5,
                    'line': {'width': 0.5, 'color': 'red'}
                })
        ],
        'layout': go.Layout(
            xaxis={
                'title': 'Date',
                'tickvals': [df_to_graph['Date'].iloc[0], df_to_graph["Date"].iloc[250], df_to_graph["Date"].iloc[500],
                             df_to_graph["Date"].iloc[750], df_to_graph["Date"].iloc[1000], df_to_graph["Date"].iloc[-1]
                             ],
            },
            yaxis={
                'title': 'Price : USD',
            },
            margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
            hovermode='closest'
        )
    }


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


@app.callback(Output('graph-learn', 'figure'), [Input('table-dropdown', 'value')])
def update_graph_learn(value_dropdown):
    if value_dropdown == 'BTC':
        csv_name = 'Gemini_BTCUSD_daily.csv'
        df_to_learn = df
    elif value_dropdown == 'ETH':
        csv_name = 'Gemini_ETHUSD_daily.csv'
        df_to_learn = dft

    # fix random seed for reproducibility
    numpy.random.seed(7)
    # load the dataset
    dataframe = pd.read_csv(csv_name, usecols=[3], engine='python')
    dataset = dataframe.values
    dataset = dataset.astype('float32')

    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # reshape dataset
    look_back = 3
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

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

    dataset = pd.DataFrame(dataset, columns=['Data'])
    trainPredictPlot = pd.DataFrame(trainPredictPlot, columns=['Train'])
    testPredictPlot = pd.DataFrame(testPredictPlot, columns=['Test'])

    return {
        'data': [go.Scatter(
            name='Actual Data',
            mode='lines',
            x=df_to_learn['Date'],
            y=dataset['Data'],
            text='Actual Data',
            marker={
                'size': 15,
                'opacity': 0.5,
                'line': {'width': 0.5, 'color': 'blue'}
            }
        ),
            go.Scatter(
                name='Training',
                mode='lines',
                x=df_to_learn['Date'],
                y=trainPredictPlot['Train'],
                text='Training',
                marker={
                    'size': 15,
                    'opacity': 0.5,
                    'line': {'width': 0.5, 'color': 'green'}
                }),

            go.Scatter(
                name='Test',
                mode='lines',
                x=df_to_learn['Date'],
                y=testPredictPlot['Test'],
                text='Test',
                marker={
                    'size': 15,
                    'opacity': 0.5,
                    'line': {'width': 0.5, 'color': 'red'}
                })
        ],
        'layout': go.Layout(
            xaxis={
                'title': 'Date',
                'tickvals': [df_to_learn['Date'].iloc[0], df_to_learn["Date"].iloc[250], df_to_learn["Date"].iloc[500],
                             df_to_learn["Date"].iloc[750], df_to_learn["Date"].iloc[1000], df_to_learn["Date"].iloc[-1]
                             ],
            },
            yaxis={
                'title': 'Price : USD',
            },
            margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
            hovermode='closest'
        )
    }


if __name__ == '__main__':
    app.run_server(debug=True)
