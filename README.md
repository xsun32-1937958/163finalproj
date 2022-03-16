README.MD

•	Download the zipped file from Kaggle (requires account):
https://www.kaggle.com/rohanrao/nifty50-stock-market-data 
Unzipping it will reveal 52 csv files inside. In addition to 51 csv files, each with market level data for individual stock, we have one more csv file that contains information and introductions for all other stock csv files. INFRATEL.csv will be dropped in our data analysis since this csv file is an empty one.  Download the uploaded code zip from gradescope (or git clone from our repo). Put them and the csv files mentioned earlier in a same folder. Now, you can run your program and start your exploration for the stock market. 

•	To set up the project, you need to install the following libraires: 
-	os
-	pandas
-	data_preprocessing
-	plotly
-	sklearn
-	numpy 
-	tensorflow
All the libraries could be installed in the ANACONDA environment.

•	We have two python programs to do the data analysis, data_preprocessing.py and research_question.py. 
To compute the three research questions, you only need to run the research_question.py. The data_preprocessing.py file converts the csv files into dataframes, drops unused columns and rows, and joins dataframes with the Date column and is imported into the research_question.py file. The research_question.py file will firstly solve the first research question. It will compute the maximum decrease and increase, and the respective happening date. Then, it plots the true v.s. predicted close prices for test dataset. Next, it solves the second research question. It will plot the stock close price for the entire IT industry, plot and compare the IT industry with other industries. Finally, it will solve the third research question. It will split the data and train a LSTM model to predict the close prices. The accuracy of this predicted prices will be computed. To check the validity of the machine learning model on another stock, run the test.py file and observe the accuracy of the new model along with the visualizations.


