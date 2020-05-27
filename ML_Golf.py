#!/usr/bin/env python
# coding: utf-8

# # Final Assignment - Big Data Programming
# ### Written and Coded by Connor Parnham

# #### Lets Import our Libraries we will use

# In[2]:


import pandas as pd
import numpy as np
import sklearn as sk
import datetime
import re
import seaborn as sns
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, scale, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from sklearn.tree import export_graphviz
get_ipython().system('conda install --yes pydot')


# #### Import our Dataset (From Kaggle)

# In[3]:


#Load in the data set and see how the data is structured
df = pd.read_csv(r'PGA_Data_Historical.csv')
df2 = df.set_index(['Player Name', 'Variable','Season'])['Value'].unstack('Variable')
df2 = df2.reset_index()
#df['Season'] = pd.DatetimeIndex(df['Date']).year
#df.drop(['Date'], axis = 1)


# #### After Looking at the excel sheet, I reduced the number of variables by taking only AVERAGES

# In[4]:


# I want to under stand what data lies in each variable and its usefulness
# Will require alot of data cleaning
var = df['Variable'].unique()
df_var = pd.DataFrame(var)
df_var.rename(columns={0:'Variable'})
#df_var.to_excel(r'C:\Users\foxnetadmin\Desktop\variables.xlsx')
var_final = pd.read_excel(r'variables.xlsx')
var_final = var_final.rename(columns={0:'Variable'})
var_final = var_final['Variable']
var_list = var_final.values.tolist()
dataset = df2[df2.columns.intersection(var_list)]
dataset = dataset.rename(columns = {'Total Money (Official and Unofficial) - (MONEY)': 'Money'})
dataset = dataset[dataset['Money'].notna()] # if they aren't making money then there is no point of having them in the data set
dataset = dataset[dataset['Birdie Average - (AVG)'].notna()] # Took the most measured variable in golf. If a player does hold this variable then there is no point of having them in data set.
data = pd.read_excel(r'dataset.xlsx')


# ### Had to Seperate the Data to fix Data Types

# In[17]:


# Had to do some data manipulation as not all data was numeric and there for needed to converted
player = data['Player Name']
int_data = data.select_dtypes(include=['int64'])
float_data = data.select_dtypes(include=['float64'])
obj_data = data.select_dtypes(include=['object'])
obj_data = obj_data.drop(['Player Name'], axis=1)
obj_data = obj_data.astype(str)
regex_o = re.compile(r'.\d$', flags=re.IGNORECASE) # regex to remove the decimals from the data. to Convert it to numeric
obj_data = obj_data.replace(regex_o,'')
obj_data = obj_data.apply(pd.to_numeric)

# Combine the data points together
data = pd.concat([player, int_data, obj_data, float_data], axis=1, sort=False)
data = data.drop(columns=['Unnamed: 0'])
corr = data.corr()
corr


# #### Dimensionality Reduction through Correlation Analysis to the target
# ##### We now have gone from 2000 variables to 110

# In[6]:


#data.to_csv(r'C:\Users\foxnetadmin\Documents\Winter Semester - BDSA\PROG8420 - Programming for Big Data\Assignments\Final Assignment\data.csv')
# after more work in excel, narrowed the variable list down to 52 using R


# ## Model Development

# In[7]:


# Ran Stepwise Selection in R to find the best model to predict 'Money'
# We now need to construct the data around the chosen variables
model_var = ["Approaches from 125-150 yards - (AVG)", "Approaches from 150-175 yards - (AVG)", "Approaches from 200-225 yards - (AVG)", "Approaches from 50-125 yards - (AVG)", "Approaches from > 100 yards - (AVG)",
"Approaches from > 200 yards - (AVG)", "3-Putt Avoidance - (%)", "Approach 125-150 yards (RTP Score) - (AVG RTP)", "Approach 150-175 yards (RTP Score) - (AVG RTP)", "Approach 175-200 yards (RTP Score) - (AVG RTP)",
"Approach 200-225 yards (RTP) - (AVG RTP)", "Approach < 125 yards (RTP Score) - (AVG RTP)", "Approach > 200 yards (RTP Score) - (AVG RTP)", "Approaches > 200 yards-Rgh (RTP) - (AVG RTP)", "Average Approach Shot Distance - (AVG)",
"Average Distance to Hole After Tee Shot - (AVG)", "Ball Speed - (AVG.)", "Birdie Average - (AVG)", "Birdie or Better Conversion Percentage - (%)", "Birdie or Better Percentage - (%)", "Bounce Back - (%)",
"Driving Pct. 240-260 (Measured) - (%)", "Driving Pct. 320+ (Measured) - (%)", "Early Par 3 Scoring Average - (AVG)", "Early Par 5 Scoring Average - (AVG)", "Fairway Approach (RTP Score) - (AVG RTP)",
"Final Round Performance - (%)", "First Tee Early Par 4 Scoring Average - (AVG)", "First Tee Early Par 5 Scoring Average - (AVG)", "First Tee Early Scoring Average - (AVG)", "First Tee Late Par 4 Scoring Average - (AVG)",
"GIR Percentage - 100+ yards - (%)", "GIR Percentage - 175-200 yards - (%)", "GIR Percentage - 200+ yards - (%)", "Going for the Green - (%)", "Good Drive Percentage - (%)", "Greens or Fringe in Regulation - (%)",
"Late Par 3 Scoring Average - (AVG)", "Late Par 5 Scoring Average - (AVG)", "One-Putt Percentage - (%)", "Par 4 Performance - (PAR 4 AVG)", "Par 5 Performance - (PAR 5 AVG)",
"Percentage of Yardage covered by Tee Shots - (AVG (%))", "Percentage of Yardage covered by Tee Shots - Par 4's - (AVG)", "Percentage of Yardage covered by Tee Shots - Par 5's - (AVG)", "Putting - Inside 10' - (% MADE)",
"Putts Per Round - (AVG)", "Scoring Average (Actual) - (AVG)", "Scoring Average - (AVG)", "Scrambling - (%)", "Scrambling > 30 yds (RTP Score) - (AVG RTP)", "Scrambling Rough (RTP Score) - (AVG RTP)", "Money"]
data1 = data[data.columns.intersection(model_var)]
data1 = data1[data1['Money'] >= 400000]


# In[8]:


# Train and Test Split (90-10 Rule)
poly = PolynomialFeatures(2)
X = data1.drop('Money', axis = 1)
y = data1['Money']/1000000
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
lm = LinearRegression()
fit = lm.fit(X_train, y_train) # training the algorithm
print(lm.intercept_)
print(lm.coef_)

# Predictions
golf_y_pred = lm.predict(X_test)
actual_pred = pd.DataFrame({'Actual': y_test, 'Predicted': golf_y_pred})
r2 = r2_score(golf_y_pred, y_test)
print("\nR-Squared of the linear model:", round(r2,2))
actual_pred


# ### Random Forest Regressor (Decision Tree)
# #### Trying a different Technique

# In[9]:


# Random Forest Regressor
rf = RandomForestRegressor(n_estimators = 1000, random_state=0)
rf.fit(X_train, y_train)

preds = rf.predict(X_test)
errors = abs(preds - y_test)
mean_error = round(np.mean(errors), 2)
mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
accuracy


# In[10]:


import pydot
tree = rf.estimators_[5]
export_graphviz(tree, out_file = 'tree.dot', feature_names = X.columns, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')


# In[11]:


rf_small = RandomForestRegressor(n_estimators=10, max_depth = 5)
rf_small.fit(X_train, y_train)
tree_small = rf_small.estimators_[5]
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = X.columns, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
graph.write_png('small_tree.png');


# In[12]:


#X_test.to_excel(r'C:\Users\foxnetadmin\Documents\Winter Semester - BDSA\PROG8420 - Programming for Big Data\Assignments\Final Assignment\data.xlsx')


# ## Developing the Main Program
# #### We want the user to be able to input some key variables to find out how much money a player can make on tour

# In[13]:


stat_name = []
stat_avg = []
var_num = []
predictions = []
# I supplied some average stats for the user, so they can have an idea of what an AVERAGE golfer could perform at
for c in range(0, len(X.columns)):
    stat_name.append(X.columns[c])
    stat_avg.append(X.iloc[:,c].mean())
    var_num.append(c)
    #print(X.columns[c],X.iloc[:,c].mean())
user_predictions = pd.DataFrame(columns = stat_name)
user_predictions.loc[len(user_predictions)] = stat_avg
#user_predictions.to_excel(r'C:\Users\foxnetadmin\Documents\Winter Semester - BDSA\PROG8420 - Programming for Big Data\Assignments\Final Assignment\user_predictions.xlsx')


# In[14]:


# When the user is ready to make predictions they can access this excel spreadsheet.
newpred = pd.read_excel(r'user_predictions.xlsx')


# In[15]:


newpred = newpred.drop('Unnamed: 0', axis = 1)
new_golf_pred = lm.predict(newpred)
# The first prediction is considered to be an averge golfer
# The second prediction is my "custom" golfer who is very strong in every aspect of the game, espcially putting and scoring.
new_golf_pred


# In[ ]:





# In[ ]:




