import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("D:\ML+PHYTHON\machine learning\supervised learning\classification\decsison tree\Iris.csv")

labelmod = LabelEncoder()
df["labeled_data"] = labelmod.fit_transform(df['Species'])

# 3. Features and Target
x = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = df["labeled_data"]

# 4. Split and Train
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

param_dict = {
    "max_depth": [1,2, 3, 5, 10],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10],
    "max_features": [None, "sqrt", "log2"],
    "max_leaf_nodes": [None, 10, 20, 50],
    "min_impurity_decrease": [0.0, 0.01, 0.05]
}

decmod = DecisionTreeClassifier() 
cv=GridSearchCV(decmod,param_grid=param_dict,cv=5,scoring='accuracy')

cv.fit(x_train,y_train)

print(cv.best_params_)
#cv.predict(x_test)


