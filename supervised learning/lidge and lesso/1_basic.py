import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

df= pd.read_csv("D:\ML+PHYTHON\machine learning\supervised learning\lidge and lesso\Fish[1].csv")

x= df[['Length1','Length2','Height','Width']]
y= df[['Weight']]
parameter={'alpha':[1,2,3,5,7,10,12,13]}

x_train,  x_test , y_train , y_test= train_test_split(x ,y ,test_size=0.2,random_state=42)

#using linear regression
modobj=LinearRegression()
modobj.fit(x_train,y_train)
prediction=modobj.predict(x_test)

lin_mse=mean_squared_error(y_test,prediction)
lin_mae=mean_absolute_error(y_test,prediction)
r2score=r2_score(y_test,prediction)
print( f"mse score= {lin_mse} and mae = {lin_mae} and r2 score= {r2score}")

#using ridge
ridgeobj=Ridge()
gridcv= GridSearchCV(ridgeobj,parameter, scoring="neg_mean_squared_error",cv=5)
gridcv.fit(x_train,y_train)

print(gridcv.best_params_)

ridgevar=gridcv.predict(x_test)
lin_mse=mean_squared_error(y_test,ridgevar)
lin_mae=mean_absolute_error(y_test,ridgevar)
r2score=r2_score(y_test,ridgevar)
print("ridgre scores")
print( f"mse score= {lin_mse} and mae = {lin_mae} and r2 score= {r2score}")

