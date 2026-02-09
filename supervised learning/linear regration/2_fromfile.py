import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Fish[1].csv")
modobj= LinearRegression()

x= df[['Length1','Length2','Height','Width']]
y= df[['Weight']]

modobj.fit(x , y)

l1= float(input("give the length1:"))
l2= float(input("give the length2:"))
h1= float(input("give the height:"))
w1= float(input("give the width1:"))
#predictedvalue = modobj.predict([[l1,l2,h1,w1]]) here is issue that we fit the machine with coluomns and now i passed plain text

new_data = pd.DataFrame([[l1, l2, h1, w1]],columns=['Length1', 'Length2', 'Height', 'Width'])

predictedvalue = modobj.predict(new_data)

print(f"the new predicted  wieght of fish is {predictedvalue}")