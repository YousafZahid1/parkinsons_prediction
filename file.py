import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC,LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
#Decision Tree, Random Forst,Gradient Boosting LEARN!


'''
#Include RandomForestClassifer  AND
# Add ROC curves,

REinforcement Learning, and explainability.'''


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

#Roc for all the models



#1) 
y_test_prob = model.predict_proba(x_test)[:,1]


fpr,tpr,threshold = roc_curve(y_test,y_test_prob)




knn_y_test_prob = knn.predict_proba(x_test)[:,1]
knn_fpr,knn_tpr,knn_threshold = roc_curve(y_test,knn_y_test_prob)
plt.plot(knn_fpr,knn_tpr,label='KNN',color="red")
#2)
svm_y_test_prob = svm.decision_function(x_test)
svm_fpr,svm_tpr,svm_threshold = roc_curve(y_test,svm_y_test_prob)
plt.plot(svm_fpr,svm_tpr,label='SVM',color="green")


plt.plot(fpr,tpr,label='Logistic Regression')
plt.plot([0,1],[0,1],'--',color="gray")


plt.legend(loc='upper right')
plt.title('Different Models ROC Curve')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
# sns.heatmap(df.corr(),annot=True,fmt='.2f',cmap='coolwarm')
plt.show()