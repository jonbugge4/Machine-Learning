import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
#THIS IS WHERE MACHINE LEARNING IS HAPPENING
knn.fit(X= data_train , y= target_train)  #Case sensititve

predicted = knn.predict(X= data_test) #dont need a target y because it predicts it
expected = target_test