# Stock-Deep-Learning
Deep Q Learning with Stock Data

The main package used is chainer, it's similar to pytorch package.

You need to install these packages in your conda environment or better use Google Collabs:

`pip install copy, numpy, pandas, chainer, matplotlib`

And make sure you have installed cudas and cudnn from nvidia (only for nvidia graphic cards) as well as an actual graphic driver.
Installation can be a bit tricky. 
Beware you need the right cudas version for a specific driver version. In my experience it can cause troubles if you install the newest driver version, that's why **I'm recommending to install the most stable, older version** like nvidia 490 and the corresponding cudas driver and cudnn package.

If you don't have an nvidia graphic card or can't seem to install cudas on your computer, just take the easy route and use GOOGLE COLABS.

# How to use this Repo

Feel free to learn, explore, modify and execute code with this repo.

A few things to mention:
- you can use GOOG.csv or aaba.us.txt or any other file. It is recommended to use the alphavantage API https://www.alphavantage.co/documentation/ or this package https://www.kaggle.com/jacksoncrow/stock-market-dataset
