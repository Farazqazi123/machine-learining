import pandas as pd
from sklearn.linear_model import LogisticRegression

x=[[1],[2],[3],[4],[5],[6]]
y=[[0],[1],[0],[1],[0],[1]]

modobj= LogisticRegression()

modobj.fit(x,y)

hours = float(input("enter the number of hours you studied = "))

predictedmarks= modobj.predict([[hours]])
print(f"based on your {hours} hours you will be {predictedmarks}")