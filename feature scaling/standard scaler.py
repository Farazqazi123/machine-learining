import pandas as pd
from sklearn.preprocessing import StandardScaler , MinMaxScaler

data = {
    'studyhours':[1,2,3,4,5,6],
    'marks':[10,20,50,70,80,13]

}

df= pd.DataFrame(data)

strdobj = StandardScaler()#createsscaler object that knows how to:calculate mean, calculate standard deviation

standardscaleddata= strdobj.fit_transform(df)

print("standard scaled data is :")
print(pd.DataFrame(standardscaleddata ,columns=['studyhours' ,'marks ']))

#Why this line?
#standardscaleddata → NumPy array (no column names)
#pd.DataFrame(...) → converts it back to DataFrame
#columns=[...] → adds meaningful column names