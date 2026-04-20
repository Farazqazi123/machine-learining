import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
df= pd.read_csv("D:\ML+PHYTHON\machine learning\supervised learning\classification\decsison tree\Iris.csv")

labelmod=LabelEncoder()
df["labbeldata"]= labelmod.fit_transform(df['Species'])
x=df[["SepalLengthCm",'SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y=df['labbeldata']


decmod=DecisionTreeClassifier()## by default make it  postprunnig technique
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

decmod.fit(x_train,y_train)

prediction = decmod.predict(x_test)

accuracy = decmod.score(x_test, y_test)
print(f"Model Accuracy (R-squared): {accuracy * 100:.2f}%")
