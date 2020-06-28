# stocks_nn_prediction
Predicting future prices of stocks from previous stock data

There are two files here. Run stocks_preprocessing_for_nn.py before running stocks_nn.py

stocks_preprocessing_for_nn.py gathers stock data of the ticker given (default: ^GSPC) and process it into labels and data using the buy_sell_hold() method.
stocks_nn.py then incorporates the labels and data into a full data set, uses majority of it to train the neural network, and the 40 most recent days to check the effectiveness of the network. Note that the check may not have sufficient buy, hold or sell data for best interpretation and you should use this at your own risk. Another 14 days of data is generated to predict 14 days into the future on stock movement

*Disclaimer: This program works on technical data alone and may not as effective as judging a stock on both its technicals and fundamentals*

Libraries needed:

pip install numpy

pip install pandas

pip install torch
