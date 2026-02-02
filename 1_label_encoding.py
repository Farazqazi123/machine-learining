import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = {
    "name": ["faraz","ahmed","thomus","vincenzo","RI","hong","seon bin jin"],
    "gender": ["male","male","female","female","male","female","male"],
    "passed": ["yes","yes","no","yes","no","yes","yes"]
}

df = pd.DataFrame(data)
df_encoder = df.copy()

le_gender = LabelEncoder()
le_passed = LabelEncoder()

df_encoder["encoded_gender"] = le_gender.fit_transform(df_encoder["gender"])
df_encoder["encoded_passed"] = le_passed.fit_transform(df_encoder["passed"])

print(df_encoder[['name', "gender", "encoded_gender", "passed", "encoded_passed"]])




