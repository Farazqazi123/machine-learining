import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

df= pd.read_csv("D:\ML+PHYTHON\machine learning\supervised learning\classification\KNN\Fish[1].csv")

df_subset= df.head(50)

print(df_subset)

le= LabelEncoder()
df_subset["encodedlabel"]=le.fit_transform(df_subset['Species'])

x=df_subset[["Weight","Length1"]]
y=df_subset["encodedlabel"]

modobj= KNeighborsClassifier(n_neighbors=8)

modobj.fit(x,y)

weight1=float(input("enter the weight of the fish"))
length2 =float(input("enter the  length of the fish"))

newdata= pd.DataFrame([[weight1,length2]], columns=["Weight",'Length1'])
prediction= modobj.predict(newdata)

if prediction[0] == 0:
    print("it is bream specie")
elif prediction[0] == 1:
    print("it is roach specie")  

    