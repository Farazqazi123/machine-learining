import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# 1. Added 'r' for the path and removed .head(100) to get all species
df = pd.read_csv("D:\ML+PHYTHON\machine learning\supervised learning\classification\decsison tree\Iris.csv")

# 2. Fixed spelling consistency for 'labeled_data'
labelmod = LabelEncoder()
df["labeled_data"] = labelmod.fit_transform(df['Species'])

# 3. Features and Target
x = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = df["labeled_data"]

# 4. Split and Train
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Note: DecisionTreeClassifier does NOT do post-pruning by default. 

decmod = DecisionTreeClassifier(max_depth=2) 
decmod.fit(x_train, y_train)

# 5. Visualization
plt.figure(figsize=(10,7))
# feature_names and class_names help make the chart readable
tree.plot_tree(decmod, 
               feature_names=x.columns, 
               class_names=labelmod.classes_.tolist(), 
               filled=True)
plt.show()