#all necessary libraries

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#Load iris DataSet
dataset = load_iris()
x = dataset['data']
y = dataset['target']

#produce random vectors

r1=np.random.rand(5)
r2=np.random.rand(5)
r3=np.random.rand(5)

# x0 = 1 for all
New_x=np.ones([150,5])
New_x[:,1:]=x


#producing new features

z1=np.ones([150,1])
z2=np.ones([150,1])
z3=np.ones([150,1])

for i1 in range(0,150) :
  z1[i1]=np.matmul(r1,New_x[i1,:])

for i2 in range(0,150) :
  z2[i2]=np.matmul(r2,New_x[i2,:])

for i3 in range(0,150) :
  z3[i3]=np.matmul(r3,New_x[i3,:])

z1=z1.reshape(150)
z2=z2.reshape(150)
z3=z3.reshape(150)

new_feachures=np.ones([150,3]);
new_feachures[:,0]=z1
new_feachures[:,1]=z2
new_feachures[:,2]=z3

X_train, X_test, y_train, y_test = train_test_split(new_feachures, y, test_size=.2)

"""#LinearRegression"""

LnR=LinearRegression()
LnR.fit(X_train,y_train)
y_pred_LnR = LnR.predict(X_test)

print("Accuracy : ", accuracy_score(y_test, y_pred_LnR.round()))
print("Confusion Matrix : \n",confusion_matrix(y_test, y_pred_LnR.round()))

"""#LogisticRegression"""

LgR=LogisticRegression(max_iter=1000)
LgR.fit(X_train,y_train)
y_pred_LgR = LgR.predict(X_test)

print("Accuracy : ", accuracy_score(y_test, y_pred_LgR.round()))
print("Confusion Matrix : \n",confusion_matrix(y_test, y_pred_LgR.round()))