import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("Fish[1].csv")
modobj= LinearRegression()

x= df[['Length1','Length2','Height','Width']]
y= df[['Weight']]

x_train,  x_test , y_train , y_test= train_test_split(x ,y ,test_size=0.2,random_state=42)
modobj.fit(x_train , y_train)


predictedvalue = modobj.predict(x_test)
print(f"the new predicted  wieght of fish is {predictedvalue}")

print("finding the accuracy :")

accuracy = modobj.score(x_test, y_test)
print(f"Model Accuracy (R-squared): {accuracy * 100:.2f}%")