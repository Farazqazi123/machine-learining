from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

x=[[1],[2],[3],[4],[5],[6]]
y=[[10],[20],[30],[40],[50],[60]]

modobj=LinearRegression()
x_train,  x_test , y_train , y_test= train_test_split(x ,y ,test_size=0.2,random_state=42)

modobj.fit(x_train,y_train)

predictedmarks= modobj.predict(x_test)
print(f"For the input {x_test[0][0]}, the predicted score is {predictedmarks[0][0]}")
#print(f"you will score {predictedmarks} marks")

# finding the accuracy
accuracy = modobj.score(x_test, y_test)
print(f"Model Accuracy (R-squared): {accuracy * 100:.2f}%")