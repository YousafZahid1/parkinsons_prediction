import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC,LinearSVC
#import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
#Decision Tree, Random Forst,Gradient Boosting LEARN


'''
Project: "Predict Parkinson’s Disease from Voice Features"

Use open UCI datasets with sklearn models to detect Parkinson’s using voice signal features.

Add ROC curves, model comparisons, and explainability.

👉 Stack: scikit-learn, seaborn, UCI Parkinson dataset, numpy, pandas
'''


df = pd.read_csv('parkinsons.csv')

X=df.drop(["status"],axis=1)
y= df["status"]

print(df.head(4))


std = StandardScaler()


x_scaled = std.fit_transform(X)


x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,test_size=0.2,random_state=42)


knn = KNeighborsClassifier(n_neighbors=5)

model = LogisticRegression()

svm = SVC(kernel = "rbf",C=1.0,gamma='scale')



model.fit(x_train,y_train)
knn.fit(x_train,y_train)
svm.fit(x_train,y_train)

y_pred = model.predict(x_test)
knn_pred = knn.predict(x_test)
svm_pred = svm.predict(x_test)
print("Logistic Regression Accuracy: ", accuracy_score(y_test, y_pred))
print("KNN Accuracy: ", accuracy_score(y_test, knn_pred))
print("SVM Accuracy: ", accuracy_score(y_test, svm_pred))
print("Confusion Matrix for Logistic Regression:\n", confusion_matrix(y_test, y_pred))

#Different Models:
# 1) 
store = cross_val_score(model,x_scaled,y,cv=5)

s = 0
for i in store:
    s+= i
print("Cross Validation Score for Logistic Regression: ", s/5)

# 2)
store = cross_val_score(knn,x_scaled,y,cv=5)

s = 0
for i in store:
    s+= i
print("Cross Validation Score for KNN: ", s/5)


# 3)

store = cross_val_score(svm,x_scaled,y,cv=5)

s = 0
for i in store:
    s+= i
print("Cross Validation Score for SVM: ", s/5)



# sns.pairplot(df,hue='status',palette='coolwarm')

sns.heatmap(df.corr(),annot=True,fmt='.2f',cmap='coolwarm')
plt.show()