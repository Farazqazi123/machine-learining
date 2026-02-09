from sklearn.linear_model import LinearRegression
import pandas as pd

x=[[1],[2],[3],[4],[5]]
y=[[10],[20],[30],[40],[50]]

modobj=LinearRegression()

modobj.fit(x,y)

hours = float(input("enter the number of hours you studied = "))

predictedmarks= modobj.predict([[hours]])

print(f"based on your {hours} hours you will score {predictedmarks} marks")