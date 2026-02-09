import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

data= {
    "hours_studied": [1,2,3,4,5,6,7,8],
    "condition": ["fail","fail","fail","fail","pass","pass","pass","pass"]
}

df=pd.DataFrame(data)

df_label= df.copy()
le= LabelEncoder()

df_label ["encoded grades"] = le.fit_transform(df_label["condition"])

modobj= LogisticRegression()

x=df_label[["hours_studied"]]
y=df_label["encoded grades"]

modobj.fit(x,y)
study=float(input("enter the number of hours you studied "))

prediction= modobj.predict([[study]])

if prediction[0] ==0:
    print("sorry bro you will be fail 😢")
else:
    print("bro you will pass dont worry 💪")    