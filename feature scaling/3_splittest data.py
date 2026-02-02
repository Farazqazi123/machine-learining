import pandas as pd
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.model_selection import train_test_split

data = {
    'studyhours':[1,2,3,4,5,6],
    'marks':[10,20,50,70,80,13]

}

df= pd.DataFrame(data)

x= df[['studyhours']]
y= df[['marks']]

x_train,  x_test , y_train , y_test= train_test_split(x ,y ,test_size=0.2,random_state=42)

print("trained data")
print (y_train)
print( "test data")
print(y_test)