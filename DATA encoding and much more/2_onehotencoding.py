import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = {
    "name": ["faraz","ahmed","thomus","vincenzo","RI","hong","seon bin jin"],
    "gender": ["male","male","female","female","male","female","male"],
    "city": ["sialkot","swat","multan","islamabad","sialkot","lahor","abbotabad"]
}

df = pd.DataFrame(data)
df_encoder = df.copy()

df_encoded=pd.get_dummies(df_encoder,columns=["city"]) #dtype = int is used to show 0 and 1 not bool
pd.set_option('display.max_columns', None)


print(df_encoded)
