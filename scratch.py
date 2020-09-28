import pandas as pd
import sklearn as sl
from sklearn import model_selection
from sklearn import tree
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
import numpy as np
#we importing the librarys
df=pd.read_csv("C:\\Users\\ersd7\\OneDrive\\Documents\\vgsales.csv")
#clean data
df.drop(['Name','Rank','Year'],axis=1,inplace=True)
df.drop(['NA_Sales','EU_Sales', 'JP_Sales', 'Other_Sales'], axis =1,inplace=True)
#we make "string" data to cartogal data
number=LabelEncoder()
df['Platform']=number.fit_transform(df['Platform'].astype('str'))
df['Genre']=number.fit_transform(df['Genre'].astype('str'))
df['Publisher']=number.fit_transform(df['Platform'].astype('str'))
columns=["Platform","Genre","Publisher","Global_Sales"]
rows=["Global_Sales"]
x=df[columns].values
y=df[rows].values
#split the data
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.30)
clf=sl.neighbors.KNeighborsRegressor()
clf.fit(x_train,y_train)
accuracy=clf.score(x_train,y_train)
print("the accuracy for training model is",accuracy*100,"%")
accuracy2=clf.score(x_test,y_test)
print("the accuracy for testing model is",accuracy2*100,"%")
print(df.head())
