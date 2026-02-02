import pandas as pd
from sklearn.preprocessing import StandardScaler , MinMaxScaler

data = {
    'studyhours':[1,2,3,4,5,6],
    'marks':[10,20,50,70,80,13]

}

df= pd.DataFrame(data)

minmaxobject= MinMaxScaler()

scaleddata= minmaxobject.fit_transform(df)

print('min max standard scaler')
print(pd.DataFrame(scaleddata , columns=['studyhours', 'marks']))