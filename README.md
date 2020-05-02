# Stock-Deep-Learning
Deep Q Learning with Stock Data

# READ CAREFUL BEFORE USING FILES

The main package used is chainer, it's similar to pytorch package.

You need to install these packages in your conda environment:

`pip install copy, numpy, pandas, chainer, matplotlib`

And make sure you have installed cudas and cudnn from nvidia (only for nvidia graphic cards) as well as an actual graphic driver.
Installation can be a bit tricky. 
Beware you need the right cudas version for a specific driver version. In my experience it can cause troubles if you install the newest driver version, that's why **I'm recommending to install the most stable, older version** like nvidia 490 and the corresponding cudas driver and cudnn package.

If you don't have an nvidia graphic card or can't seem to install cudas on your computer, just take the easy route and use GOOGLE COLABS.

# How to use this Repo

Feel free to learn, explore, modify and execute code with this repo.

A few things to mention:
- "QLearningModel(stock_daily_chart).model" is the model, that is used in the ipynb file.
- Use the ipynb file "DeepQLearning_Stocks.ipynb" to see what kind of results you should have when running the main file  	"_DeepQLearningAgent_v1,1.py"
- The features of the	_DeepQLearningAgent_v1,1.py are the last 90 previous close to current close % changes. You can change them to whatever indicator you like by using libraries such as the "ta" lib
- you can choose between Dueling Double Q learning and normal Double Q learning
- in default settings, the "QLearningModel(stock_daily_chart).model" is used. Results can vary from the ipynb file result, because it will choose different training and test data slices (intervals in form of quarters) for every run. To prevent this, use a seed and save the seed. Turn on MAKE_MODEL variable to start training
- if you have more features in the sample space or more deep layers, then implement autosaves of your model, because in the current version it will only save the model after all epochs are finished (which you can set up too).
- you can use GOOG.csv or aaba.us.txt or any other file. It is recommended to use the alphavantage API https://www.alphavantage.co/documentation/
