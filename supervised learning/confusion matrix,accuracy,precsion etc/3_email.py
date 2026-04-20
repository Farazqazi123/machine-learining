import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix,ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer ##used to convert texts to numbers

df=pd.read_csv("D:\ML+PHYTHON\machine learning\supervised learning\confusion matrix,accuracy,precsion etc\emails.csv")

col=df['text']
textobj=TfidfVectorizer()
x=textobj.fit_transform(col)
y=df["spam"]

x_train,  x_test , y_train , y_test= train_test_split(x ,y ,test_size=0.2,random_state=42)

modobj= LogisticRegression()

modobj.fit(x_train,y_train)

predicted=modobj.predict(x_test)

print(f"accuracy = {accuracy_score(y_test,predicted)*100}%")
print(f"precision= {precision_score(y_test,predicted)}")
print(f"recallscore = {recall_score(y_test,predicted)}")
print("confusion matrix is")

cm= confusion_matrix(y_test,predicted)
print(cm)

#visualizing it
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
