import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

x=[
    [180,7],  #weight and size
    [200,7.5],
    [250,8],
    [300 , 9],
    [400 , 10],
    [420, 10.5],
    [500, 11]

]

y=[0,0,0,1,1,1,1]

model= KNeighborsClassifier(n_neighbors=3)

model.fit(x,y)

weight=float(input("enter the weight of the fruit"))
size =float(input("enter the size of the fruit"))

prediction=model.predict([[weight,size]])[0]

if prediction==0:
    print("it is apple")

elif prediction==1:
    print("it is orange")
