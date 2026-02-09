import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

x=[[1],[2],[3],[4],[5],[6]]
#y=[[0],[1],[0],[1],[0],[1]]
y=[0,0,0,1,1,1]

modobj= LogisticRegression()

x_train,  x_test , y_train , y_test= train_test_split(x ,y ,test_size=0.2,random_state=42)
modobj.fit(x_train,y_train)

prediction = modobj.predict(x_test)

print(f"For the input {x_test}, the predicted grade is {prediction[0]}")

# finding the  accuracy
accuracy = modobj.score(x_test, y_test)
print(f"Model Accuracy (R-squared): {accuracy * 100:.2f}%")


